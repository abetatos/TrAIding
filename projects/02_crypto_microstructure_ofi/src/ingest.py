"""
Binance Vision data ingestion for crypto microstructure (Project 02).

Source: https://data.binance.vision  (public S3 dumps, no API key)
  futures/um/daily/bookTicker/<SYMBOL>/<SYMBOL>-bookTicker-YYYY-MM-DD.zip
  futures/um/daily/aggTrades/<SYMBOL>/<SYMBOL>-aggTrades-YYYY-MM-DD.zip

Top-of-book (bookTicker) is only published for a bounded window per symbol; aggTrades is
continuous. See the availability table in the project README. Requested ranges are clamped
to known availability and stored partitioned by day so ingestion is resumable.

Usage:
    python src/ingest.py                                   # 2-day dev sample, both datasets
    python src/ingest.py --start 2024-03-01 --end 2024-03-30
    python src/ingest.py --dataset agg_trades --start 2023-05-16 --end 2024-03-30
    python src/ingest.py --verify                          # also check SHA256 checksums
"""
import argparse
import hashlib
import io
import logging
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Allow running from project root or from src/
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.data_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BINANCE_VISION = "https://data.binance.vision"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Canonical column names (Binance Vision CSVs carry a header on recent dates, but not always —
# the reader below is header-robust and forces these names regardless).
COLUMNS = {
    "book_ticker": [
        "update_id", "bid_px", "bid_qty", "ask_px", "ask_qty",
        "transaction_time", "event_time",
    ],
    "agg_trades": [
        "agg_id", "price", "qty", "first_id", "last_id",
        "transact_time", "is_buyer_maker",
    ],
}

# Binance Vision dataset folder names
VISION_DATASET = {"book_ticker": "bookTicker", "agg_trades": "aggTrades"}

# Known availability windows (probed 2026-06-18). aggTrades is effectively unbounded.
# Only bookTicker is the binding constraint; other symbols may differ — verified for BTCUSDT.
AVAILABILITY = {
    "book_ticker": {"BTCUSDT": (date(2023, 5, 16), date(2024, 3, 30))},
    "agg_trades": {},  # continuous; no clamp
}


# ── Download ─────────────────────────────────────────────────────────────────

def _daily_url(dataset: str, symbol: str, day: date) -> str:
    ds = VISION_DATASET[dataset]
    return (
        f"{BINANCE_VISION}/data/futures/um/daily/{ds}/{symbol}/"
        f"{symbol}-{ds}-{day.isoformat()}.zip"
    )


def _download(url: str) -> bytes | None:
    """Stream a file into memory. Returns None on 404 (missing day)."""
    resp = requests.get(url, timeout=120, stream=True)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
    return buf.getvalue()


def _verify_checksum(zip_bytes: bytes, url: str) -> bool:
    """Check SHA256 against the sibling .CHECKSUM file (format: '<sha256>  <filename>')."""
    resp = requests.get(url + ".CHECKSUM", timeout=30)
    if resp.status_code != 200:
        log.warning("No checksum available for %s", url.rsplit("/", 1)[-1])
        return True  # don't fail ingestion if Binance didn't publish one
    expected = resp.text.strip().split()[0]
    actual = hashlib.sha256(zip_bytes).hexdigest()
    if actual != expected:
        log.error("Checksum MISMATCH for %s", url.rsplit("/", 1)[-1])
        return False
    return True


# ── Parse ────────────────────────────────────────────────────────────────────

def _read_csv_zip(zip_bytes: bytes, dataset: str) -> pd.DataFrame:
    """Read the single CSV inside a Binance Vision zip, header-robust."""
    names = COLUMNS[dataset]
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        fname = z.namelist()[0]
        with z.open(fname) as f:
            first = f.readline().decode("utf-8", "ignore").split(",")[0].strip()
        # Header present iff the first field is not a number (numeric id ⇒ no header)
        has_header = not first.lstrip("-").isdigit()
        with z.open(fname) as f:
            df = pd.read_csv(
                f,
                header=0 if has_header else None,
                names=None if has_header else names,
            )
    df.columns = names  # force canonical names regardless of source header text
    return df


def _normalize_book_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["transaction_time"], unit="ms", utc=True)
    for c in ("bid_px", "bid_qty", "ask_px", "ask_qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df["spread"] = df["ask_px"] - df["bid_px"]
    return df[["ts", "update_id", "bid_px", "bid_qty", "ask_px", "ask_qty", "mid", "spread"]]


def _normalize_agg_trades(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    # is_buyer_maker == True  ⇒ aggressor is the seller ⇒ sign -1 (downtick pressure)
    is_maker = df["is_buyer_maker"].astype(str).str.lower().eq("true")
    df["sign"] = (~is_maker).astype(int) * 2 - 1  # buy aggressor +1, sell aggressor -1
    return df[["ts", "agg_id", "price", "qty", "sign"]]


NORMALIZE = {"book_ticker": _normalize_book_ticker, "agg_trades": _normalize_agg_trades}


# ── Availability / date range ────────────────────────────────────────────────

def _clamp_range(dataset: str, symbol: str, start: date, end: date) -> tuple[date, date]:
    window = AVAILABILITY.get(dataset, {}).get(symbol)
    if window is None:
        return start, end
    lo, hi = window
    c_start, c_end = max(start, lo), min(end, hi)
    if (c_start, c_end) != (start, end):
        log.warning(
            "%s/%s availability is %s..%s — clamped request to %s..%s",
            dataset, symbol, lo, hi, c_start, c_end,
        )
    return c_start, c_end


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def ingest(
    dataset: str,
    symbol: str,
    start: date,
    end: date,
    verify: bool = False,
    overwrite: bool = False,
) -> None:
    start, end = _clamp_range(dataset, symbol, start, end)
    if start > end:
        log.warning("Empty range for %s/%s after clamping — skipping", dataset, symbol)
        return

    out_dir = DATA_DIR / dataset / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    days = list(_daterange(start, end))

    for day in tqdm(days, desc=f"{dataset}/{symbol}"):
        out_path = out_dir / f"{day.isoformat()}.parquet"
        if out_path.exists() and not overwrite:
            continue

        url = _daily_url(dataset, symbol, day)
        zip_bytes = _download(url)
        if zip_bytes is None:
            log.warning("Missing (404): %s", url.rsplit("/", 1)[-1])
            continue
        if verify and not _verify_checksum(zip_bytes, url):
            continue

        df = NORMALIZE[dataset](_read_csv_zip(zip_bytes, dataset))
        save_parquet(df, out_path, compression="zstd")

    log.info("Done %s/%s → %s  (%d day-partitions on disk)",
             dataset, symbol, out_dir, len(list(out_dir.glob("*.parquet"))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance Vision microstructure data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--dataset", default="both",
        choices=["book_ticker", "agg_trades", "both"],
    )
    # Default: tiny 2-day sample at the end of the bookTicker availability window
    parser.add_argument("--start", default="2024-03-29", help="YYYY-MM-DD")
    parser.add_argument("--end", default="2024-03-30", help="YYYY-MM-DD")
    parser.add_argument("--verify", action="store_true", help="verify SHA256 checksums")
    parser.add_argument("--overwrite", action="store_true", help="re-download existing days")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    datasets = ["book_ticker", "agg_trades"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        ingest(ds, args.symbol, start, end, verify=args.verify, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
