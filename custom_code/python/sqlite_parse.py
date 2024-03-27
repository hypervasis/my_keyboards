import argparse
import sqlite3
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd

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
parser.add_argument("--avg", type=int, default=10, help="rolling average in days")

args = parser.parse_args()
layout = "us-colemak" if args.layout == "colemak" else "us"
limitdays = args.lastdays if args.lastdays else 100000


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


get_first_appearence_query = f"""select
  date(min(timeStamp))
from
  (
    select
      *
    from
      lessons
    where
      layout = "{layout}"
    order by
      timeStamp desc
    limit
      {limitdays}
  )
"""

get_daily_summary_query = f"""select
  *
from
  (
    select
      date(timeStamp) date,
      avg(speed) / 5 wpm,
      avg(speed) / 60 cps,
      sum(length) chars_typed,
      100 - sum(errors) * 100.0 / sum(length) accuracy_perc,
      sum(time) / 1000.0 / 60 practise_time,
      avg(time) / 1000.0 time_per_lesson,
      count(*) lessons
    from
      lessons
    where
      layout = "{layout}"
    group by
      date(timeStamp)
    order by
      date(timeStamp) desc
    limit
      {limitdays}
  )
order by
  date"""

get_detailed_query = f"""select
  date(timeStamp) date,
  speed / 5 wpm,
  speed / 60 cps,
  length chars_typed,
  100 - errors * 100.0 / length accuracy_perc,
  time / 1000.0 / 60 practise_time,
  time / 1000.0 time_per_lesson
from
  lessons
where
  layout = "{layout}"
  and timestamp > DATETIME('now', '-{limitdays} days')
order by
  timeStamp
"""

get_daily_chars_stats_query = f"""select
  *
from(
    select
      date(timeStamp) date,
      count(*) as lessons,
      sum(spc_hitCount) spc_times,
      12000 / avg(spc_timeToType) spc_hitime,
      100 - sum(spc_missCount) * 100.0 / sum(spc_hitCount) spc_accuracy,
      sum(a_hitCount) a_times,
      12000 / avg(a_timeToType) a_hitime,
      100 - sum(a_missCount) * 100.0 / sum(a_hitCount) a_accuracy,
      sum(b_hitCount) b_times,
      12000 / avg(b_timeToType) b_hitime,
      100 - sum(b_missCount) * 100.0 / sum(b_hitCount) b_accuracy,
      sum(c_hitCount) c_times,
      12000 / avg(c_timeToType) c_hitime,
      100 - sum(c_missCount) * 100.0 / sum(c_hitCount) c_accuracy,
      sum(d_hitCount) d_times,
      12000 / avg(d_timeToType) d_hitime,
      100 - sum(d_missCount) * 100.0 / sum(d_hitCount) d_accuracy,
      sum(e_hitCount) e_times,
      12000 / avg(e_timeToType) e_hitime,
      100 - sum(e_missCount) * 100.0 / sum(e_hitCount) e_accuracy,
      sum(f_hitCount) f_times,
      12000 / avg(f_timeToType) f_hitime,
      100 - sum(f_missCount) * 100.0 / sum(f_hitCount) f_accuracy,
      sum(g_hitCount) g_times,
      12000 / avg(g_timeToType) g_hitime,
      100 - sum(g_missCount) * 100.0 / sum(g_hitCount) g_accuracy,
      sum(h_hitCount) h_times,
      12000 / avg(h_timeToType) h_hitime,
      100 - sum(h_missCount) * 100.0 / sum(h_hitCount) h_accuracy,
      sum(i_hitCount) i_times,
      12000 / avg(i_timeToType) i_hitime,
      100 - sum(i_missCount) * 100.0 / sum(i_hitCount) i_accuracy,
      sum(j_hitCount) j_times,
      12000 / avg(j_timeToType) j_hitime,
      100 - sum(j_missCount) * 100.0 / sum(j_hitCount) j_accuracy,
      sum(k_hitCount) k_times,
      12000 / avg(k_timeToType) k_hitime,
      100 - sum(k_missCount) * 100.0 / sum(k_hitCount) k_accuracy,
      sum(l_hitCount) l_times,
      12000 / avg(l_timeToType) l_hitime,
      100 - sum(l_missCount) * 100.0 / sum(l_hitCount) l_accuracy,
      sum(m_hitCount) m_times,
      12000 / avg(m_timeToType) m_hitime,
      100 - sum(m_missCount) * 100.0 / sum(m_hitCount) m_accuracy,
      sum(n_hitCount) n_times,
      12000 / avg(n_timeToType) n_hitime,
      100 - sum(n_missCount) * 100.0 / sum(n_hitCount) n_accuracy,
      sum(o_hitCount) o_times,
      12000 / avg(o_timeToType) o_hitime,
      100 - sum(o_missCount) * 100.0 / sum(o_hitCount) o_accuracy,
      sum(p_hitCount) p_times,
      12000 / avg(p_timeToType) p_hitime,
      100 - sum(p_missCount) * 100.0 / sum(p_hitCount) p_accuracy,
      sum(q_hitCount) q_times,
      12000 / avg(q_timeToType) q_hitime,
      100 - sum(q_missCount) * 100.0 / sum(q_hitCount) q_accuracy,
      sum(r_hitCount) r_times,
      12000 / avg(r_timeToType) r_hitime,
      100 - sum(r_missCount) * 100.0 / sum(r_hitCount) r_accuracy,
      sum(s_hitCount) s_times,
      12000 / avg(s_timeToType) s_hitime,
      100 - sum(s_missCount) * 100.0 / sum(s_hitCount) s_accuracy,
      sum(t_hitCount) t_times,
      12000 / avg(t_timeToType) t_hitime,
      100 - sum(t_missCount) * 100.0 / sum(t_hitCount) t_accuracy,
      sum(u_hitCount) u_times,
      12000 / avg(u_timeToType) u_hitime,
      100 - sum(u_missCount) * 100.0 / sum(u_hitCount) u_accuracy,
      sum(v_hitCount) v_times,
      12000 / avg(v_timeToType) v_hitime,
      100 - sum(v_missCount) * 100.0 / sum(v_hitCount) v_accuracy,
      sum(w_hitCount) w_times,
      12000 / avg(w_timeToType) w_hitime,
      100 - sum(w_missCount) * 100.0 / sum(w_hitCount) w_accuracy,
      sum(x_hitCount) x_times,
      12000 / avg(x_timeToType) x_hitime,
      100 - sum(x_missCount) * 100.0 / sum(x_hitCount) x_accuracy,
      sum(y_hitCount) y_times,
      12000 / avg(y_timeToType) y_hitime,
      100 - sum(y_missCount) * 100.0 / sum(y_hitCount) y_accuracy,
      sum(z_hitCount) z_times,
      12000 / avg(z_timeToType) z_hitime,
      100 - sum(z_missCount) * 100.0 / sum(z_hitCount) z_accuracy
    from
      lessons
    where
      layout = '{layout}'
    group by
      date(timeStamp)
    order by
      timeStamp desc
    limit
      {limitdays}
  )
order by
  date
"""

total_stats_query = f"""select
  round(avg(wpm), 2) wpm,
  round(avg(accuracy_perc), 2) accuracy_perc,
  round(sum(practise_time), 2) practise_time,
  sum(chars_typed) chars_typed,
  round(sum(chars_typed) / 614044.0, 3) odyssey_perc
from
  (
    select
      date(timeStamp) date,
      avg(speed) / 5 wpm,
      sum(length) chars_typed,
      100 - sum(errors) * 100.0 / sum(length) accuracy_perc,
      sum(time) / 1000.0 / 60 practise_time
    from
      lessons
    where
      layout = "{layout}"
    group by
      date(timeStamp)
    order by
      date(timeStamp) desc
    limit
      {limitdays}
  )
"""


def get_first_appearance(conn, letter):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    q = f"{get_first_appearence_query} where {letter}_hitCount is not null"
    # print(q)
    cur.execute(q)
    row = cur.fetchone()
    # print(letter, row)
    return row[0]


def get_total_stats(conn):
    cur = conn.cursor()
    cur.execute(total_stats_query)
    row = cur.fetchone()
    # print(row)
    return row


conn = None
try:
    conn = sqlite3.connect("/tmp/data.db")
except sqlite3.Error as e:
    print(f"Cannot open sqlite db [data.db]. Error:{e}")
    exit()

appearance = [
    (
        f"{chr(97+i)}",
        datetime.strptime(get_first_appearance(conn, f"{chr(97+i)}"), "%Y-%m-%d"),
    )
    for i in range(26)
    if get_first_appearance(conn, f"{chr(97+i)}")
]
appearance.insert(
    0, ("spc", datetime.strptime(get_first_appearance(conn, "spc"), "%Y-%m-%d"))
)
appearance = sorted(appearance, key=lambda x: (x[1]))

# print(appearance)

appearance_merged = []
prevDate = None
currInd = 0
letterInd = 0
right_side = "jluymneiokh"
first = True
for app in appearance:
    sideInd = "r" if app[0] in right_side else "" if app[0] == "spc" else "l"
    # print("dealing with:", app)
    if prevDate == app[1]:
        appearance_merged[currInd - 1] = [
            prevDate,
            # f"{appearance_merged[currInd - 1][1]}\n{app[0]}{letterInd}{sideInd}"
            # if first
            # else f"{appearance_merged[currInd - 1][1]}\n{app[0]}\n{letterInd}\n{sideInd}",
            f"{appearance_merged[currInd - 1][1]}\n{app[0]}{letterInd}{sideInd}",
        ]
    else:
        first = False if prevDate else True
        appearance_merged.append(
            [
                app[1],
                f"{app[0]}{letterInd}{sideInd}"
                if first
                else f"{app[0]}\n{letterInd}\n{sideInd}",
            ]
        )
        currInd += 1
        prevDate = app[1]
    letterInd += 1
    letterInd = 0 if letterInd > 9 else letterInd

# Check for really long (many lines) letter sequences
for app in appearance_merged:
    lines = app[1].split("\n")
    if len(lines) > 7:
        app[1] = "\n".join(lines[0:2] + ["..."] + lines[-2:])

df = pd.read_sql_query(get_daily_chars_stats_query, conn)
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df = df.fillna(value=np.nan)

dates_num = mdates.date2num(df["date"])

plt.style.use("default")
dateFmt = mdates.DateFormatter("%b-%d")


#         _       _           _          _
#    __ _| |_ __ | |__   __ _| |__   ___| |_
#   / _` | | '_ \| '_ \ / _` | '_ \ / _ \ __|
#  | (_| | | |_) | | | | (_| | |_) |  __/ |_
#   \__,_|_| .__/|_| |_|\__,_|_.__/ \___|\__|
#          |_|

with plt.style.context("Solarize_Light2"):
    # hcArr = np.array(hit)
    # figure, axis = plt.subplots(27, 3, sharex=True, sharey="col", figsize=(18, 64))
    figure, axis = plt.subplots(27, 3, sharex=True, figsize=(24, 64))
    # figure, axis = plt.subplots(26, sharex=True, sharey=True, figsize=(12, 40))

    for letter in range(27):
        char = f"{chr(97 + letter)}" if letter < 26 else "spc"
        graphs = [
            {"datacol": df[f"{char}_hitime"], "color": 0, "ax": 0, "miny": None},
            {"datacol": df[f"{char}_accuracy"], "color": 2, "ax": 1, "miny": 80},
            {"datacol": df[f"{char}_times"], "color": 3, "ax": 2, "miny": None},
        ]

        for g in graphs:
            datacol = g["datacol"]
            ax = axis[letter, g["ax"]]
            idx = np.isfinite(df["date"]) & np.isfinite(datacol)
            if not (~idx).all():
                z = np.polyfit(dates_num[idx], datacol[idx], 1)
                p = np.poly1d(z)
                ax.scatter(
                    dates_num,
                    datacol,
                    color=plt.rcParams["axes.prop_cycle"].by_key()["color"][g["color"]],
                )
                ax.plot(
                    dates_num[idx],
                    p(dates_num)[idx],
                    color="orange",
                )
                ax.plot(
                    dates_num[idx],
                    datacol.ewm(span=args.avg, adjust=True).mean()[idx],
                    color="red",
                )
                for iv, vert_line in enumerate(appearance_merged):
                    ax.axvline(
                        x=vert_line[0],
                        ymax=0.72 if len(vert_line[1].split("\n")) < 4 else 0.44,
                        linewidth=1.0,
                        linestyle=(0, (1, 4)),
                        c="#073642",
                    )
                    blended_trans = transforms.blended_transform_factory(
                        ax.transData, ax.transAxes
                    )
                    ax.text(
                        vert_line[0],
                        0.98,
                        vert_line[1],
                        horizontalalignment="center",
                        verticalalignment="top",
                        transform=blended_trans,
                    )
                    if g["miny"]:
                        if char == "spc":
                            ax.set_ylim([97, 100])
                        else:
                            ax.set_ylim([g["miny"], 100])

    col_headers = ["wpm", "accuracy(%)", "chars typed"]
    row_headers = [chr(97 + i) for i in range(26)]
    row_headers.append("spc")

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


#   _ __  _ __ ___   __ _ _ __ ___  ___ ___
#  | '_ \| '__/ _ \ / _` | '__/ _ \/ __/ __|
#  | |_) | | | (_) | (_| | | |  __/\__ \__ \
#  | .__/|_|  \___/ \__, |_|  \___||___/___/
#  |_|              |___/

df = pd.read_sql_query(get_daily_summary_query, conn)
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

df_detail = pd.read_sql_query(get_detailed_query, conn)
df_detail["date"] = pd.to_datetime(df_detail["date"], format="ISO8601")
dates_num_detail = mdates.date2num(df_detail["date"])

df = df.fillna(value=np.nan)


def calc_score(row):
    ptime_mult = 1.0
    if row["practise_time"] < 30.0:
        ptime_mult = 1.0 - (30.0 - row["practise_time"]) / 100.0
    elif row["practise_time"] > 31.0:
        ptime_mult = 1.0 - (row["practise_time"] - 31.0) / 100.0

    return row["wpm"] * row["accuracy_perc"] / 100.0 * ptime_mult


df["score"] = df.apply(calc_score, axis=1)

with plt.style.context("Solarize_Light2"):
    # hcArr = np.array(hit)

    # figure = plt.figure(constrained_layout=True)
    # gs = GridSpec(2, 4, figure=figure)
    figure, axis = plt.subplot_mosaic(
        "HHIIJJKKLL;AAAAAAAAAA;BBBBBCCCCC;DDDDDEEEEE;FFFFFGGGGG",
        height_ratios=[0.1, 1, 1, 1, 1],
        # sharex=True,
        constrained_layout=True,
        figsize=(12, 18),
    )
    # figure, axis = plt.subplots(26, sharex=True, sharey=True, figsize=(12, 40))

    dataCols = [
        {
            "datacol": "wpm",
            "title": "wpm",
            "color": 0,
            "ax": "A",
            "ylabel": "wpm",
            "details": True,
        },
        {
            "datacol": "accuracy_perc",
            "title": "accuracy",
            "color": 2,
            "ax": "B",
            "ylabel": "percentage",
            "details": True,
        },
        {
            "datacol": "practise_time",
            "title": "practice time",
            "color": 3,
            "ax": "C",
            "ylabel": "mins",
            "details": False,
        },
        {
            "datacol": "chars_typed",
            "title": "chars typed",
            "color": 4,
            "ax": "D",
            "ylabel": "chars",
            "details": False,
        },
        {
            "datacol": "score",
            "title": "score",
            "color": 6,
            "ax": "E",
            "ylabel": "points",
            "details": False,
        },
        {
            "datacol": "lessons",
            "title": "lessons",
            "color": 7,
            "ax": "F",
            "ylabel": "lessons",
            "details": False,
        },
        {
            "datacol": "time_per_lesson",
            "title": "Time/Lesson",
            "color": 0,
            "ax": "G",
            "ylabel": "sec",
            "details": True,
        },
    ]
    for i in dataCols:
        dataCol = df[i["datacol"]]
        ax = axis[i["ax"]]

        z = np.polyfit(dates_num, dataCol, 1)
        p = np.poly1d(z)

        # full details
        if i["details"]:
            dataCol_detail = df_detail[i["datacol"]]
            ax.scatter(
                dates_num_detail,
                dataCol_detail,
                color="r",
                s=0.5,
            )
        # daily summary
        ax.scatter(
            dates_num,
            dataCol,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i["color"]],
            s=16.0,
        )
        ax.plot(dates_num, p(dates_num), color="orange")
        ax.plot(dates_num, dataCol.ewm(span=args.avg, adjust=True).mean(), color="red")

        ax.set_title(i["title"])
        for iv, vert_line in enumerate(appearance_merged):
            blended_trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes
            )
            ax.axvline(
                x=vert_line[0],
                ymax=0.88 if len(vert_line[1].split("\n")) < 4 else 0.75,
                linewidth=1.0,
                linestyle=(0, (1, 4)),
                c="#073642",
            )
            ax.text(
                vert_line[0],
                # axis[ax].get_ylim()[1],
                0.99,
                vert_line[1],
                horizontalalignment="center",
                verticalalignment="top",
                size=8,
                transform=blended_trans,
            )
        ax.xaxis.set_major_formatter(dateFmt)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8, labelright=True)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylabel(i["ylabel"])

    total_stats = get_total_stats(conn)
    # fix the numbers we get back
    # total_stats[1] = str(timedelta(minutes=total_stats[1]))

    stats_header = [
        "wpm\n",
        "accuracy\n",
        "practice\ntime\n",
        "chars\ntyped\n",
        "odysseys\ntyped\n",
    ]
    for i, stat in enumerate(total_stats):
        ax = chr(ord("H") + i)
        if i == 2:
            stat = str(timedelta(minutes=stat))
            stat = stat.replace(",", ",\n")
            # Remove a potential decimal on seconds
            stat = stat.split(".")[0]

        axis[ax].text(
            0.5,
            0.5,
            f"{stats_header[i]}{stat}",
            ha="center",
            va="bottom",
            color="#073642",
            size=0.3 * 72,
            bbox=dict(boxstyle="round4", fc="#EEE8D5", ec="#073642"),
        )
        axis[ax].axis("off")

    # figure.autofmt_xdate()
    figure.tight_layout()
    plt.savefig(f"progress-{args.layout}.png")
