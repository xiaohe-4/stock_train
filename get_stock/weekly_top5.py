"""
提取2026年3月起每周涨幅前五的股票
涨幅计算：(周末开盘价 - 周初开盘价) / 周初开盘价 × 100%
结果保存为 data/weekly_top5_stocks.md
"""

import pandas as pd

# 读取数据
df = pd.read_csv('data/stock_data.csv')
df['日期'] = pd.to_datetime(df['日期'])

df = df[df['日期'] >= '2024-01-02'].copy()

# 按ISO周编号分组
df['year_week'] = (
    df['日期'].dt.isocalendar().year.astype(str)
    + '-W'
    + df['日期'].dt.isocalendar().week.astype(str).str.zfill(2)
)

# 计算每只股票每周的开盘价涨幅
results = []
for (stock, week), group in df.groupby(['股票代码', 'year_week']):
    group = group.sort_values('日期')
    first_open = group.iloc[0]['开盘']
    last_open = group.iloc[-1]['开盘']
    if first_open > 0:
        change_pct = (last_open - first_open) / first_open * 100
    else:
        change_pct = 0
    results.append({
        '股票代码': stock,
        '周': week,
        '周起始日期': group.iloc[0]['日期'].strftime('%Y-%m-%d'),
        '周结束日期': group.iloc[-1]['日期'].strftime('%Y-%m-%d'),
        '周初开盘价': first_open,
        '周末开盘价': last_open,
        '涨幅(%)': round(change_pct, 2),
    })

res_df = pd.DataFrame(results)

# 每周取涨幅前5
top5 = (
    res_df.sort_values(['周', '涨幅(%)'], ascending=[True, False])
    .groupby('周')
    .head(5)
)

# 生成 Markdown
weeks = sorted(res_df['周'].unique())
data_start = df['日期'].min().strftime('%Y-%m-%d')
data_end = df['日期'].max().strftime('%Y-%m-%d')
stock_count = df['股票代码'].nunique()

lines = [
    f'# {data_start[:4]}年{int(data_start[5:7])}月起每周涨幅前五股票',
    '',
    f'> 数据范围：{data_start} 至 {data_end}',
    '> 涨幅计算：(周末开盘价 - 周初开盘价) / 周初开盘价 × 100%',
    f'> 股票数量：{stock_count}只',
    '',
]

for week in weeks:
    week_data = top5[top5['周'] == week].sort_values('涨幅(%)', ascending=False)
    if week_data.empty:
        continue
    start = week_data.iloc[0]['周起始日期']
    end = week_data.iloc[0]['周结束日期']
    lines.append(f'## {week}（{start} ~ {end}）')
    lines.append('')
    lines.append('| 排名 | 股票代码 | 周初开盘价 | 周末开盘价 | 涨幅(%) |')
    lines.append('|:----:|:--------:|:----------:|:----------:|:-------:|')
    for rank, (_, row) in enumerate(week_data.iterrows(), 1):
        code = str(int(row['股票代码'])).zfill(6)
        lines.append(
            f'| {rank} | {code} | {row["周初开盘价"]:.2f} | {row["周末开盘价"]:.2f} | {row["涨幅(%)"]:+.2f}% |'
        )
    lines.append('')

md_content = '\n'.join(lines)
output_path = 'data/weekly_top5_stocks.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f'已保存 {output_path}，共 {len(weeks)} 周')
