# # dicts look like:
# {
#     "layout": "us",
#     "lessonType": "guided",
#     "timeStamp": "2023-03-18T17:51:10.000Z",
#     "length": 123,
#     "time": 35204,
#     "errors": 1,
#     "speed": 209.6352687194637,
#     "histogram": [{
#         "codePoint": 32,
#         "hitCount": 23,
#         "missCount": 0,
#         "timeToType": 243
#     }, {
#         "codePoint": 101,
#         "hitCount": 37,
#         "missCount": 0,
#         "timeToType": 264
#     }, {
#         "codePoint": 105,
#         "hitCount": 10,
#         "missCount": 0,
#         "timeToType": 343
#     }, {
#         "codePoint": 108,
#         "hitCount": 4,
#         "missCount": 0,
#         "timeToType": 655
#     }, {
#         "codePoint": 110,
#         "hitCount": 24,
#         "missCount": 0,
#         "timeToType": 295
#     }, {
#         "codePoint": 114,
#         "hitCount": 12,
#         "missCount": 0,
#         "timeToType": 257
#     }, {
#         "codePoint": 116,
#         "hitCount": 13,
#         "missCount": 1,
#         "timeToType": 298
#     }]
#  }


import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Parse exported data from keybr.com.",
)
parser.add_argument(
    "--layout",
    type=str,
    choices=["qwerty", "colemak"],
    help="which layout to parse",
    default="colemak",
)
parser.add_argument(
    "--lastdays", type=int, default=0, help="parse the last x days only"
)

args = parser.parse_args()

plt.style.use("default")
dateFmt = mdates.DateFormatter("%b-%d")


def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs,
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


def extractLetter(h: list, letter: str, key: str):
    for i in h:
        if i.get("codePoint") == ord(letter):
            return i.get(key, 0)

    return np.nan


def keep_item(i, layout: str, days_threshold: timedelta) -> bool:
    if i["layout"] != layout:
        return False
    if days_threshold:
        return (
            datetime.now() - datetime.fromisoformat(i["timeStamp"][:-1])
            < days_threshold
        )
    return True


# layouts (from keybr): Qwerty: "us", Colemak: "us-colemak"
layout = "us-colemak" if args.layout == "colemak" else "us"

data = json.load(Path("typing-data.json").open())

days_threshold = timedelta(days=args.lastdays) if args.lastdays else None
data = [x for x in data if keep_item(x, layout, days_threshold)]


dates = [datetime.fromisoformat(i["timeStamp"][:-1]) for i in data]
dates_num = mdates.date2num(dates)
dnArr = np.array(dates_num)


hitcounts = {}
misses = {}
times = {}

for letter in range(26):
    hitcounts[chr(97 + letter)] = np.array(
        [extractLetter(i["histogram"], chr(97 + letter), "hitCount") for i in data]
    )
    misses[chr(97 + letter)] = np.array(
        [extractLetter(i["histogram"], chr(97 + letter), "missCount") for i in data]
    )
    times[chr(97 + letter)] = np.array(
        [extractLetter(i["histogram"], chr(97 + letter), "timeToType") for i in data]
    )

first_seen = {}
for i in data:
    hist = i["histogram"]
    for h in hist:
        cp = h["codePoint"]
        if cp not in first_seen:
            first_seen[cp] = datetime.fromisoformat(i["timeStamp"][:-1])

print(f"Firstseen: {first_seen}")

first_seen_by_date = []
for k, v in first_seen.items():
    first_seen_by_date.append({"date": v, "char": chr(k) if chr(k) != " " else "_"})

first_seen_by_date = sorted(first_seen_by_date, key=lambda d: d["date"])

print(f"first_seen_by_date: {first_seen_by_date}")

prevEntry = None

first_seen_by_merged = {}
for combo in first_seen_by_date:
    if prevEntry and combo.get("date") - prevEntry < timedelta(days=1):
        first_seen_by_merged[prevEntry] = (
            first_seen_by_merged[prevEntry] + "," + combo["char"]
        )
    else:
        first_seen_by_merged[combo["date"]] = combo["char"]
        prevEntry = combo["date"]

print(f"first_seen_by_merged: {first_seen_by_merged}")
exit()

with plt.style.context("Solarize_Light2"):
    # hcArr = np.array(hit)
    figure, axis = plt.subplots(26, 3, sharex=True, sharey="col", figsize=(18, 64))
    # figure, axis = plt.subplots(26, sharex=True, sharey=True, figsize=(12, 40))

    for letter in range(26):
        idx = np.isfinite(dnArr) & np.isfinite(hitcounts[chr(97 + letter)])
        if not (~idx).all():
            z = np.polyfit(dnArr[idx], hitcounts[chr(97 + letter)][idx], 1)
            p = np.poly1d(z)
            axis[letter, 0].scatter(
                dates,
                hitcounts[chr(97 + letter)],
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            )
            axis[letter, 0].plot(dates_num[idx], p(dates_num)[idx], color="red")
            for vert_line in first_seen_by_merged.keys():
                axis[letter, 0].axvline(x=vert_line, linewidth=1.0, linestyle="--")
                axis[letter, 0].text(
                    vert_line + timedelta(hours=3),
                    axis[letter, 0].get_ylim()[1],
                    first_seen_by_merged[vert_line],
                )

        idx = np.isfinite(dnArr) & np.isfinite(misses[chr(97 + letter)])
        if not (~idx).all():
            z = np.polyfit(dnArr[idx], misses[chr(97 + letter)][idx], 1)
            p = np.poly1d(z)
            axis[letter, 1].scatter(
                dates,
                misses[chr(97 + letter)],
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
            )
            axis[letter, 1].plot(dates_num[idx], p(dates_num)[idx], color="red")
            for vert_line in first_seen_by_merged.keys():
                axis[letter, 1].axvline(x=vert_line, linewidth=1.0, linestyle="--")
                axis[letter, 1].text(
                    vert_line + timedelta(hours=3),
                    axis[letter, 1].get_ylim()[1],
                    first_seen_by_merged[vert_line],
                )

        idx = np.isfinite(dnArr) & np.isfinite(times[chr(97 + letter)])
        if not (~idx).all():
            z = np.polyfit(dates_num[idx], times[chr(97 + letter)][idx], 1)
            p = np.poly1d(z)
            axis[letter, 2].scatter(
                dates,
                times[chr(97 + letter)],
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][3],
            )
            axis[letter, 2].plot(dates_num[idx], p(dates_num)[idx], color="red")
            axis[letter, 2].set_ylim([0, 1000])
            for vert_line in first_seen_by_merged.keys():
                axis[letter, 2].axvline(x=vert_line, linewidth=1.0, linestyle="--")
                axis[letter, 2].text(
                    vert_line + timedelta(hours=3),
                    axis[letter, 2].get_ylim()[1],
                    first_seen_by_merged[vert_line],
                )

    # plt.xticks(rotation=45)

    col_headers = ["hitCounts", "misses", "time to hit"]
    row_headers = [chr(97 + i) for i in range(26)]

    add_headers(
        figure,
        col_headers=col_headers,
        row_headers=row_headers,
        rotate_row_headers=False,
        **dict(fontfamily="monospace", fontweight="bold", fontsize="x-large"),
    )
    plt.gca().xaxis.set_major_formatter(dateFmt)
    figure.autofmt_xdate()
    figure.tight_layout()
    plt.savefig(f"alphabet-{args.layout}.png")


length = np.array([i["length"] for i in data])
time = np.array([i["time"] / 1000 for i in data])
errors = np.array([i["errors"] for i in data])
speed = np.array([i["speed"] / 5 for i in data])  # chars/min


with plt.style.context("Solarize_Light2"):
    # hcArr = np.array(hit)
    figure, axis = plt.subplots(2, 2, sharex=True, figsize=(18, 11))
    # figure, axis = plt.subplots(26, sharex=True, sharey=True, figsize=(12, 40))

    z = np.polyfit(dnArr, length, 1)
    p = np.poly1d(z)
    axis[0, 0].scatter(
        dates, length, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    )
    axis[0, 0].plot(dates_num, p(dates_num), color="red")
    axis[0, 0].set_title("Chars/lesson")
    for vert_line in first_seen_by_merged.keys():
        axis[0, 0].axvline(x=vert_line, linewidth=1.0, linestyle="--")
        axis[0, 0].text(
            vert_line + timedelta(hours=3),
            axis[0, 0].get_ylim()[1],
            first_seen_by_merged[vert_line],
        )

    z = np.polyfit(dnArr, time, 1)
    p = np.poly1d(z)
    axis[0, 1].scatter(
        dates, time, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    )
    axis[0, 1].plot(dates_num, p(dates_num), color="red")
    axis[0, 1].set_title("Time/lesson (secs)")
    for vert_line in first_seen_by_merged.keys():
        axis[0, 1].axvline(x=vert_line, linewidth=1.0, linestyle="--")
        axis[0, 1].text(
            vert_line + timedelta(hours=3),
            axis[0, 1].get_ylim()[1],
            first_seen_by_merged[vert_line],
        )

    z = np.polyfit(dnArr, errors, 1)
    p = np.poly1d(z)
    axis[1, 0].scatter(
        dates, errors, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][3]
    )
    axis[1, 0].plot(dates_num, p(dates_num), color="red")
    axis[1, 0].set_title("Errors/lesson")
    for vert_line in first_seen_by_merged.keys():
        axis[1, 0].axvline(x=vert_line, linewidth=1.0, linestyle="--")
        axis[1, 0].text(
            vert_line + timedelta(hours=3),
            axis[1, 0].get_ylim()[1],
            first_seen_by_merged[vert_line],
        )

    z = np.polyfit(dnArr, speed, 1)
    p = np.poly1d(z)
    axis[1, 1].scatter(
        dates, speed, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][4]
    )
    axis[1, 1].plot(dates_num, p(dates_num), color="red")
    axis[1, 1].set_title("Speed (wpm)")
    for vert_line in first_seen_by_merged.keys():
        axis[1, 1].axvline(x=vert_line, linewidth=1.0, linestyle="--")
        axis[1, 1].text(
            vert_line + timedelta(hours=3),
            axis[1, 1].get_ylim()[1],
            first_seen_by_merged[vert_line],
        )

    plt.gca().xaxis.set_major_formatter(dateFmt)
    figure.autofmt_xdate()
    figure.tight_layout()
    # plt.legend()
    plt.savefig(f"progress-{args.layout}.png")
