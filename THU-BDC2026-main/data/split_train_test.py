import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="按日期区间将股票数据切分为 train.csv 和 test.csv；"
		"默认按最新数据自动划分（最后 N 个交易日为测试集）"
	)
	parser.add_argument(
		"--input",
		type=str,
		default="data/stock_data.csv",
		help="原始数据文件路径，默认 data/stock_data.csv",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="data",
		help="输出目录，默认 data",
	)
	parser.add_argument(
		"--auto",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="是否根据原始数据自动划分（默认开启；可用 --no-auto 关闭）",
	)
	parser.add_argument(
		"--test-trading-days",
		type=int,
		default=5,
		help="自动划分时，测试集使用最近多少个交易日，默认 5",
	)
	parser.add_argument(
		"--train-start",
		type=str,
		default=None,
		help="训练集开始日期；自动模式下默认取原始数据最早交易日",
	)
	parser.add_argument(
		"--train-end",
		type=str,
		default=None,
		help="训练集结束日期；自动模式下默认为测试集前一交易日",
	)
	parser.add_argument(
		"--test-start",
		type=str,
		default=None,
		help="测试集开始日期；自动模式下默认取最近 N 个交易日的首日",
	)
	parser.add_argument(
		"--test-end",
		type=str,
		default=None,
		help="测试集结束日期；自动模式下默认取最近交易日",
	)
	return parser.parse_args()


def _to_timestamp(date_str: str, name: str) -> pd.Timestamp:
	ts = pd.to_datetime(date_str, errors="coerce")
	if pd.isna(ts):
		raise ValueError(f"参数 {name} 的日期格式无效: {date_str}")
	return ts.normalize()


def _validate_columns(df: pd.DataFrame) -> None:
	required = {"股票代码", "日期"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"输入文件缺少必要列: {sorted(missing)}")


def _filter_by_date(
	df: pd.DataFrame,
	start_date: pd.Timestamp,
	end_date: pd.Timestamp,
) -> pd.DataFrame:
	if start_date > end_date:
		raise ValueError(f"开始日期晚于结束日期: {start_date.date()} > {end_date.date()}")

	mask = (df["日期"] >= start_date) & (df["日期"] <= end_date)
	out = df.loc[mask].copy()
	out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
	out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
	return out


def _resolve_split_dates(
	trading_days: list[pd.Timestamp],
	args: argparse.Namespace,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
	if len(trading_days) == 0:
		raise ValueError("原始数据中没有任何有效交易日")

	source_min = trading_days[0]
	source_max = trading_days[-1]

	if args.auto:
		n = args.test_trading_days
		if n <= 0:
			raise ValueError("--test-trading-days 必须为正整数")
		if len(trading_days) <= n:
			raise ValueError(
				f"交易日数量({len(trading_days)})不足以划出 {n} 天测试集"
			)

		test_days = trading_days[-n:]
		train_days = trading_days[:-n]
		train_start = (
			_to_timestamp(args.train_start, "--train-start")
			if args.train_start
			else source_min
		)
		train_end = (
			_to_timestamp(args.train_end, "--train-end")
			if args.train_end
			else train_days[-1]
		)
		test_start = (
			_to_timestamp(args.test_start, "--test-start")
			if args.test_start
			else test_days[0]
		)
		test_end = (
			_to_timestamp(args.test_end, "--test-end")
			if args.test_end
			else test_days[-1]
		)
		print(
			f"自动划分: 最近 {n} 个交易日作为测试集 "
			f"({test_start.date()} ~ {test_end.date()})"
		)
		return train_start, train_end, test_start, test_end

	# 手动模式：必须显式给出四个日期
	required = {
		"--train-start": args.train_start,
		"--train-end": args.train_end,
		"--test-start": args.test_start,
		"--test-end": args.test_end,
	}
	missing = [name for name, value in required.items() if not value]
	if missing:
		raise ValueError(
			"关闭自动划分时必须提供完整日期参数: " + ", ".join(missing)
		)

	return (
		_to_timestamp(args.train_start, "--train-start"),
		_to_timestamp(args.train_end, "--train-end"),
		_to_timestamp(args.test_start, "--test-start"),
		_to_timestamp(args.test_end, "--test-end"),
	)


def main() -> None:
	args = parse_args()

	input_path = Path(args.input)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(input_path)
	_validate_columns(df)

	df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
	if df["日期"].isna().any():
		bad_rows = int(df["日期"].isna().sum())
		raise ValueError(f"原始数据中存在无法解析的日期，共 {bad_rows} 行")

	trading_days = sorted(pd.to_datetime(df["日期"].dropna().unique()))
	source_min_date = trading_days[0].date()
	source_max_date = trading_days[-1].date()

	train_start, train_end, test_start, test_end = _resolve_split_dates(
		trading_days, args
	)

	if train_end >= test_start:
		raise ValueError(
			f"训练集结束日({train_end.date()})必须早于测试集开始日({test_start.date()})"
		)

	train_df = _filter_by_date(df, train_start, train_end)
	test_df = _filter_by_date(df, test_start, test_end)

	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"

	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)

	print(f"原始数据日期范围: {source_min_date} ~ {source_max_date}")
	print(f"训练集: {train_path}，共 {len(train_df)} 行，股票数 {train_df['股票代码'].nunique()}")
	print(f"测试集: {test_path}，共 {len(test_df)} 行，股票数 {test_df['股票代码'].nunique()}")
	print(
		f"训练集日期范围: {train_start.date()} ~ {train_end.date()} | "
		f"测试集日期范围: {test_start.date()} ~ {test_end.date()}"
	)
	if train_df.empty or test_df.empty:
		print("警告: 训练集或测试集为空，请检查日期范围是否与原始数据重叠。")


if __name__ == "__main__":
	main()
