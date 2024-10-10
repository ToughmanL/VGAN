#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   textgridprocess.py
@Time    :   2022/11/02 20:05:46
@Author  :   lxk 
@Version :   1.0
@Contact :   xk.liu@siat.ac.cn
@License :   (C)Copyright 2022-2025, lxk&AISML
@Desc    :   None
'''

from collections import namedtuple
# from language_util import _SILENCE

Entry = namedtuple("Entry", ["start",
                             "stop",
                             "name",
                             "tier",
                             "interval_index"])


def _find_tiers(interval_lines, tier_lines, tiers):
    tier_pairs = zip(tier_lines, tiers)
    tier_pairs = iter(tier_pairs)
    cur_tline, cur_tier = next(tier_pairs)
    next_tline, next_tier = next(tier_pairs, (None, None))
    tiers = []
    for il in interval_lines:
        if next_tline is not None and il > next_tline:
            cur_tline, cur_tier = next_tline, next_tier
            next_tline, next_tier = next(tier_pairs, (None, None))           
        tiers.append(cur_tier)
    return tiers 


def _read(f):
    return [x.strip() for x in f.readlines()]


def write_csv(textgrid_list, filename=None, sep=",", header=True, save_gaps=False, meta=True):
    """
    Writes a list of textgrid dictionaries to a csv file.
    If no filename is specified, csv is printed to standard out.
    """
    columns = list(Entry._fields)
    if filename:
        f = open(filename, "w")
    if header:
        hline = sep.join(columns)
        if filename:
            f.write(hline + "\n")
        else:
            print(hline)
    for entry in textgrid_list:
        row = sep.join(str(x) for x in list(entry))
        if filename:
            f.write(row + "\n")
        else:
            print(row)
    if filename:
        f.flush()
        f.close()
    if meta:
        with open(filename + ".meta", "w") as metaf:
            metaf.write("""---\nunits: s\ndatatype: 1002\n""")


def _build_entry(i, content, tier):
    """
    takes the ith line that begin an interval and returns
    a dictionary of values
    """
    start = _get_float_val(content[i + 1])  # addition is cheap typechecking
    if content[i].startswith("intervals ["):
        offset = 1
    else:
        offset = 0  # for "point" objects
    stop = _get_float_val(content[i + 1 + offset])
    label = _get_str_val(content[i + 2 + offset])
    return Entry(start=start, stop=stop, name=label, tier=tier, interval_index=i)


def _get_float_val(string):
    """
    returns the last word in a string as a float
    """
    return float(string.split()[-1])


def _get_str_val(string):
    """
    returns the last item in quotes from a string
    """
    return string.split('"')[-2]


def read_textgrid(filename, tierName='phones'):
    """
    Reads a TextGrid file into a dictionary object
    each dictionary has the following keys:
    "start"
    "stop"
    "name"
    "tier"

    Points and intervals use the same format, 
    but the value for "start" and "stop" are the same
    """
    if isinstance(filename, str):
        with open(filename, "r", encoding='utf-8') as f:
            content = _read(f)
    elif hasattr(filename, "readlines"):
        content = _read(filename)
    else:
        raise TypeError("filename must be a string or a readable buffer")

    interval_lines = [i for i, line in enumerate(content)
                      if line.startswith("intervals [")
                      or line.startswith("points [")]
    tier_lines = []
    tiers = []
    for i, line in enumerate(content):
        if line.startswith("name ="):
            tier_lines.append(i)
            tiers.append(line.split('"')[-2])

    for i, line in enumerate(content):
        if line.startswith("xmax = "):
            time_array = line.split()
            time_array = [item for item in filter(lambda x: x.strip() != '', time_array)]
            duration = float(time_array[-1])
            break

    interval_tiers = _find_tiers(interval_lines, tier_lines, tiers)
    assert len(interval_lines) == len(interval_tiers)
    adjust_list = [_build_entry(i, content, t) for i, t in zip(interval_lines, interval_tiers) if t == tierName]
    return adjust_list




if __name__ == "__main__":
    #textgrid2csv()
    grid_file = '/mnt/shareEEx/liuxiaokang/data/MSDM/labeled_data/Control/N_10008_F/N_F_10008_G1_task1_1.TextGrid'
    csv_file = 'N_F_10008_G1_task1_1.csv'
    tgrid = read_textgrid(grid_file, 'TEXT')
    error = read_textgrid(grid_file, 'ERROR')
    print(tgrid)
    print(error)
    write_csv(tgrid, csv_file, sep=' ', header=False, meta=False)