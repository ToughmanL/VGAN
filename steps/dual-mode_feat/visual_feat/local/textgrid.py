##############################################################################
# The MIT License (MIT)
# 
# Copyright (c) 2016 Kyler Brown
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################

#!/usr/bin/python

from collections import namedtuple
# from language_util import _SILENCE

Entry = namedtuple("Entry", ["start",
                             "stop",
                             "name",
                             "tier",
                             "interval_index"])

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
        try:
            with open(filename, "r", encoding='utf-8') as f:
                content = _read(f)
        except:
            try:
                with open(filename, "r", encoding='utf-16') as f:
                    content = _read(f)
            except:
                print('TextGrid file style error')
                exit(-1)
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
    # adjust_list = _adjust_Entry([_build_entry(i, content, t) for i, t in zip(interval_lines, interval_tiers)
    #         if t == "Phon"], duration)
    adjust_list = [_build_entry(i, content, t) for i, t in zip(interval_lines, interval_tiers) if t == tierName]
    return adjust_list

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

def _adjust_Entry(textgrid_list, _duration):
    """adjust the textgrid_list, expecially the start and end:
    Start:
        label[0]=="" and lable[1]=="sp"  -----   merge, label[0]=="sil"
        label[0]<>"" and label[0]<>"sp"  -----   add, label[0]==”sil"
        label[0]=="" and label[1]<>"sp"  -----   change, label[0]=="sil"
        #label[0]=="sp" and label[1]<>"sp"-----   change, label[0]=="sil"
    End:
        label[-1]=="" and lable[-2]=="sp"  -----   merge, label[-1]=="sil"
        label[-1]<>"" and label[-1]<>"sp"  -----   add, label[-1]==”sil"
        label[-1]=="" and label[-2]<>"sp"  -----   change, label[-1]=="sil"
        #label[-1]=="sp" and label[-2]<>"sp"-----   change, label[-1]=="sil"
    """
    textgrid_list_new = textgrid_list
    if len(textgrid_list_new) > 1:
        if textgrid_list_new[0].name == "" and textgrid_list_new[1].name in _SILENCE:
            textgrid_list_new[1] = textgrid_list_new[1]._replace(name="sil")._replace(start=0)
            textgrid_list_new.pop(0)
        if textgrid_list_new[0].name != "" and textgrid_list_new[0].name not in _SILENCE:
            textgrid_list_new.insert(0, Entry(start=0, stop=0.001, name="sil", tier="phones"))
            textgrid_list_new[1] = textgrid_list_new[1]._replace(start=0.001)
        if textgrid_list_new[0].name in ["", "sp"] and textgrid_list_new[1].name not in _SILENCE and textgrid_list_new[1].name != "":
            textgrid_list_new[0] = textgrid_list_new[0]._replace(name="sil")

        if textgrid_list_new[-1].name == "" and textgrid_list_new[-2].name in _SILENCE:
            textgrid_list_new[-2] = textgrid_list_new[-2]._replace(name="sil")._replace(stop=_duration)
            textgrid_list_new.pop()
        if textgrid_list_new[-1].name != "" and textgrid_list_new[-1].name not in _SILENCE:
            tmp_time = textgrid_list_new[-1].stop - 0.001
            textgrid_list_new.append(Entry(start=tmp_time, stop=_duration, name="sil", tier="phones"))
            textgrid_list_new[-2] = textgrid_list_new[-2]._replace(stop=tmp_time)
        if textgrid_list_new[-1].name in ["", "sp"] and textgrid_list_new[-2].name not in _SILENCE and textgrid_list_new[-2].name != "":
            textgrid_list_new[-1] = textgrid_list_new[-1]._replace(name="sil")
    return textgrid_list_new


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
        # if entry.name or save_gaps:  # skip unlabeled intervals
        #     row = sep.join(str(x) for x in list(entry))
        #     if filename:
        #         f.write(row + "\n")
        #     else:
        #         print(row)
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


def textgrid2csv():
    import argparse
    parser = argparse.ArgumentParser(description="convert a TextGrid file to a CSV.")
    parser.add_argument("TextGrid",
                        help="a TextGrid file to process")
    parser.add_argument("-o", "--output", help="(optional) outputfile")
    parser.add_argument("--sep", help="separator to use in CSV output",
                        default=",")
    parser.add_argument("--noheader", help="no header for the CSV",
                        action="store_false")
    parser.add_argument("--savegaps", help="preserves intervals with no label",
            action="store_true")
    args = parser.parse_args()
    tgrid = read_textgrid(args.TextGrid)
    write_csv(tgrid, args.output, args.sep, args.noheader, args.savegaps)

def write_entry2textgrid(tiers, duration, new_gridpath):
    text_line = []
    text_line.append("File type = \"ooTextFile\"")
    text_line.append("Object class = \"TextGrid\"")
    text_line.append("")

    text_line.append("xmin = 0 ")
    text_line.append("xmax = {} ".format(str(duration)))
    text_line.append("tiers? <exists> ")
    text_line.append("size = {} ".format(str(int(len(tiers)))))
    text_line.append("item []: ")

    for i in range(len(tiers)):
        text_line.append("\titem [{}]:".format(str(i+1)))
        text_line.append("\t\tclass = \"IntervalTier\" ")
        text_line.append("\t\tname = \"{}\"".format(tiers[i][0].tier))
        text_line.append("\t\txmin = 0 ")
        text_line.append("\t\txmax = {}".format(str(duration)))
        text_line.append("\t\tintervals: size = {} ".format(str(len(tiers[i]))))
        for j in range(len(tiers[i])):
            text_line.append("\t\tintervals [{}]:".format(j+1))
            text_line.append("\t\t\txmin = {} ".format(tiers[i][j].start))
            text_line.append("\t\t\txmax = {}".format(tiers[i][j].stop))
            text_line.append("\t\t\ttext = \"{}\"".format(tiers[i][j].name))
    with open(new_gridpath, 'w') as fp:
        for line in text_line:
            fp.write(line + '\n')


if __name__ == "__main__":
    #textgrid2csv()
    grid_file = "/mnt/shareEx/caodi/data/20230605/Control/N_10001_F/N_F_10001_G1_task1_1.TextGrid"
    csv_file = '/mnt/shareEx/caodi/code/video_only/test/data/tg2csv.csv'
    tgrid = read_textgrid(grid_file, 'refphone')
    # print(tgrid)
    write_csv(tgrid, csv_file, sep=' ', header=False, meta=False)