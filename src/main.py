import analyzer
import database
import fetcher


def main():
    print(f"Is CUDA available? {analyzer.is_cudable()}")
    print("--------------------")

    fetcher.fetch_daily_quotes_of_nikkei225_to_db()


if __name__ == "__main__":
    main()
