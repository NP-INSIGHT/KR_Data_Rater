"""
KR Data Rater - CLI
데이터 기반 종목 기술적 분석 (OHLCV + MA → LLM)
"""

import sys
import argparse

import engine


def cmd_stocks(args):
    """종목 분석 실행."""
    provider = args.provider or engine.LLM_PROVIDER

    if args.watchlist:
        names = engine.get_active_watchlist(args.list)
        if not names:
            print("워치리스트에 활성 종목이 없습니다.")
            return 1
    elif args.tickers:
        names = args.tickers
    else:
        print("--tickers 또는 --watchlist를 지정하세요.")
        return 1

    # Multi-temperature 결정
    if args.temperatures:
        temperatures = args.temperatures
    elif args.multi_temp or engine.MULTI_TEMP_ENABLED:
        temperatures = engine.MULTI_TEMPERATURES
    else:
        temperatures = None

    def log(msg):
        print(msg, flush=True)

    result = engine.run_stock_analysis(names, provider, log_callback=log, temperatures=temperatures)

    # Word 보고서 생성
    if result.get("a_rated"):
        try:
            docx_path = engine.save_results_docx(result, "stocks")
            log(f"\nWord 보고서: {docx_path}")
        except Exception as e:
            log(f"\nWord 보고서 생성 실패: {e}")

    return 0


def cmd_watchlist(args):
    """워치리스트 관리."""
    if args.add:
        added = engine.add_to_watchlist(args.add, args.list)
        print(f"{added}개 종목 추가됨")
    elif args.remove:
        removed = engine.remove_from_watchlist(args.remove, args.list)
        print(f"{removed}개 종목 삭제됨")
    elif args.show:
        stocks = engine.load_watchlist(args.list)
        if not stocks:
            print("워치리스트가 비어있습니다.")
        else:
            for s in stocks:
                status = "O" if s.get("active", True) else "X"
                print(f"  [{status}] {s['name']}")
    return 0


def cmd_refresh_cache(args):
    """종목 캐시 갱신."""
    count = engine.refresh_ticker_cache()
    print(f"ticker_cache.json 갱신 완료: {count}개 종목")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="KR Data Rater - 데이터 기반 종목 기술적 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # stocks
    p_stocks = sub.add_parser("stocks", help="종목 분석")
    p_stocks.add_argument("--tickers", nargs="+", help="분석할 종목명")
    p_stocks.add_argument("--watchlist", action="store_true", help="워치리스트 종목 분석")
    p_stocks.add_argument("--list", default=None, help="워치리스트 이름")
    p_stocks.add_argument("--provider", choices=["claude", "gemini"], help="LLM provider")
    p_stocks.add_argument("--multi-temp", action="store_true", help="Multi-temperature analysis (config values)")
    p_stocks.add_argument("--temperatures", nargs="+", type=float, help="Temperature values (e.g. 0 0.3 0.7)")

    # watchlist
    p_wl = sub.add_parser("watchlist", help="워치리스트 관리")
    p_wl.add_argument("--add", nargs="+", help="종목 추가")
    p_wl.add_argument("--remove", nargs="+", help="종목 삭제")
    p_wl.add_argument("--show", action="store_true", help="종목 목록 표시")
    p_wl.add_argument("--list", default=None, help="리스트 이름")

    # refresh-cache
    sub.add_parser("refresh-cache", help="종목 캐시 갱신")

    args = parser.parse_args()

    if args.command == "stocks":
        return cmd_stocks(args)
    elif args.command == "watchlist":
        return cmd_watchlist(args)
    elif args.command == "refresh-cache":
        return cmd_refresh_cache(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
