# from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataProc():
  def __init__(self, csv_path):
    self.data = pd.read_csv(csv_path)
    self.feat_names = self.data.keys()
    self.normal_data = self.data[self.data.Person.str.startswith('N')]
    self.dysart_data = self.data[self.data.Person.str.startswith('S')]
    self.normal_f_data = self.normal_data[self.normal_data.Person.str.endswith('F')]
    self.normal_m_data = self.normal_data[self.normal_data.Person.str.endswith('M')]
    self.dysart_f_data = self.dysart_data[self.dysart_data.Person.str.endswith('F')]
    self.dysart_m_data = self.dysart_data[self.dysart_data.Person.str.endswith('M')]
    self.persons_data = {}
    self.vowels_data = {}

  def _person_division(self):
    uniqs = set(self.data['Person'])
    for uniq in uniqs:
      self.persons_data[uniq] = self.data[self.data.Person == uniq]
  
  def _vowel_dividion(self):
    # uniqs = set(self.data['Vowel'])
    uniqs = ['a','o','e','i','u','v']
    for uniq in uniqs:
      # self.vowels_data[uniq] = self.data[self.data.Vowel == uniq]
      self.vowels_data[uniq] = self.data[self.data.TEXT.str.contains(uniq)]
  
  def _hist(self, dir, vowel_name, feat_name, norm_feat_dict, dysa_feat_dict):
    norm_label = norm_feat_dict['label']
    norm_feat_data = norm_feat_dict['data']
    dysa_label = dysa_feat_dict['label']
    dysa_feat_data = dysa_feat_dict['data']
    # sns.distplot(norm_feat_data,bins=20,kde=False,hist_kws={"color":"steelblue"},label="normal")
    sns.distplot(norm_feat_data,bins=20,hist=True,hist_kws={"color":"steelblue"},kde_kws={"color":"blue","linestyle":"-"},norm_hist=True,label=norm_label)
    # sns.distplot(dysa_feat_data,bins=20,kde=False,hist_kws={"color":"purple"},label="dysarthria")
    sns.distplot(dysa_feat_data,bins=20,hist=True,hist_kws={"color":"purple"},kde_kws={"color":"red","linestyle":"--"},norm_hist=True,label=dysa_label)
    plt.title("{} {} Histogram".format(feat_name, vowel_name))
    plt.legend()
    # plt.show()
    fname = dir + '/histogram_{}_{}.png'.format(feat_name, vowel_name)
    plt.savefig(fname, dpi=75, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)
    plt.clf()
  
  def SN_distribution(self, fname):
    fig,ax = plt.subplots()
    ax.bar(self.normal_data.Person.value_counts().index, self.normal_data.Person.value_counts().values, width=0.6, color='blue', label='normal')
    ax.bar(self.dysart_data.Person.value_counts().index, self.dysart_data.Person.value_counts().values, width=0.6, color='red', label='dysrthria')
    plt.xticks(rotation=90, fontsize=6)
    plt.legend()
    # plt.show()
    plt.savefig(fname, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)

  def all_vowels_feat_hist(self,dir,beg,end):
    for col_i in range(beg,end):
      clo_name = self.feat_names[col_i]
      norm_feat_data = {'label':'normal','data':self.normal_data[clo_name]}
      dysa_feat_data = {'label':'dysarthria', 'data':self.dysart_data[clo_name]}
      norm_f_feat_data = {'label':'normal_female', 'data':self.normal_f_data[clo_name]}
      norm_m_feat_data = {'label':'normal_male', 'data':self.normal_m_data[clo_name]}
      dysa_f_feat_data = {'label':'dysarthria_female', 'data':self.dysart_f_data[clo_name]}
      dysa_m_feat_data = {'label':'dysarthria_male', 'data':self.dysart_m_data[clo_name]}
      self._hist(dir, 'ConPac_', clo_name, norm_feat_data, dysa_feat_data)
      self._hist(dir, 'Con_MF', clo_name, norm_m_feat_data, norm_f_feat_data)
      self._hist(dir, 'Pac_MF', clo_name, dysa_m_feat_data, dysa_f_feat_data)
  
  def single_vowel_feat_hist(self,dir,beg,end):
    self._vowel_dividion()
    for vowel, feat_data in self.vowels_data.items():
      for col_i in range(beg, end):
        clo_name = self.feat_names[col_i]
        norm_feat_data = feat_data[feat_data.Person.str.startswith('N')][clo_name]
        dysa_feat_data = feat_data[feat_data.Person.str.startswith('S')][clo_name]
        self._hist(dir, vowel, clo_name, norm_feat_data, dysa_feat_data)
  
  def missing_proportion(self):
    # fill_data = self.data.fillna(value=None)
    # result = ((self.data.notnull().sum())/self.data.shape[0]).sort_values(ascending=False).map(lambda x:"{:.2%}".format(x))
    all_bar_result = ((self.data.isnull().sum())/self.data.shape[0]).sort_values(ascending=False)
    nor_bar_result = ((self.normal_data.isnull().sum())/self.normal_data.shape[0]).sort_values(ascending=False)
    dys_bar_result = ((self.dysart_data.isnull().sum())/self.dysart_data.shape[0]).sort_values(ascending=False)
    fig,ax = plt.subplots()
    floats = [float(x) for x in list(range(39))]
    ax.bar([i-0.3 for i in floats], all_bar_result.values, width=0.3, align='edge',color='black', label='all')
    ax.bar(floats, nor_bar_result.values, width=0.3, align='center', color='blue', label='normal')
    ax.bar([i for i in floats], dys_bar_result.values, width=0.3, align='edge', color='red', label='dysart')
    plt.xticks(list(range(39)),self.feat_names,rotation=90, fontsize=6)
    plt.legend()
    plt.show()
    # print(bar_result)

if __name__ == "__main__":
  # csv_path = "/tmp/LXKDATA/result_intermediate/acoustic_loop_feats_sel_1w.csv"
  csv_path = "/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/regression_train/tmp/droped_data.csv"
  pic_path = "tmp/feat_pics/2w_feat/"
  dataproc = DataProc(csv_path)
  # dataproc.SN_distribution(pic_path + "test.png")
  dataproc.all_vowels_feat_hist(pic_path, 1, 91)
  dataproc.single_vowel_feat_hist(pic_path, 1, 91)
  # dataproc.missing_proportion()