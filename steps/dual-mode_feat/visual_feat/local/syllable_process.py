'''
处理info中与syllable有关的数据

'''
import pandas as pd
class SyllableProcess():
  def __init__(self, Syllable_path) -> None:
    self.Syllable_path = Syllable_path

# 读取单元音韵母的音节数据：返回text和syllable
  def Get_All_Syllable(self):
    mono_vowel_dic = dict()
    with open(self.Syllable_path, 'r', encoding='utf-8') as f:
      for line in f.readlines():
        line = line.strip('\n').split(" ")
        text = line[0]
        syllable = "".join(line[1:])
        mono_vowel_dic.update({text:syllable})
    return mono_vowel_dic
   

if __name__ == "__main__":
  Syllable_path="/mnt/shareEx/caodi/code/video_only/info/Mono_Vowel_Syllables.txt"

  syllable_test=SyllableProcess(Syllable_path)
  mono_vowel_dic = syllable_test.Get_All_Syllable()

  print(list(mono_vowel_dic.keys())[0])
  print(list(mono_vowel_dic.values())[0])