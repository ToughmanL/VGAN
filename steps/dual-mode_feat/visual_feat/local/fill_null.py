class Fill:
  #填充坐标
  def fill_coord(self, frame_coordinate):
    new = {}
    if len(frame_coordinate):
      for ii in range(0, 106):
        new['coordinate_' + str(ii)] = str(frame_coordinate[ii][0]) + ',' + str(frame_coordinate[ii][1])
    else:
      for ii in range(0, 106):
        new['coordinate_' + str(ii)] = 'nan,nan'
    return new
  #填充归一化后的帧
  def fill_null_frame(self, aligned_list):
    i = 1
    while (i < len(aligned_list)):
      if aligned_list[i] == []:
        aligned_list[i] = aligned_list[i - 1]
      i += 1
    return aligned_list

  # #填充归一化后的帧
  # def fill_null_frame(self, aligned_list):
  #   i = 1
  #   continue_count_list = []
  #   # 从第2个找到倒数第二个
  #   while (i < len(aligned_list) - 1):
  #     # 找到第一个不为缺失值的表格
  #     if aligned_list[i] == []:
  #       # 以i为起点继续向后找连续缺失值
  #       continue_count = 1
  #       for null_index in range(i + 1, len(aligned_list)):
  #         if aligned_list[null_index] == []:
  #           continue_count += 1
  #         else:
  #           continue_count_list.append(continue_count)
  #           break
  #       # 填充缺失值：如果只有一个缺失值
  #       if continue_count == 1:
  #         before = aligned_list[i - 1]
  #         after = aligned_list[i + 1]
  #         aligned_list[i] = (before + after) / 2
  #       if continue_count > 1:
  #         # 如果后面全是空缺值
  #         if i + continue_count -1 == len(aligned_list) -1 :
  #           for k in range(i, i + continue_count - 1):
  #             aligned_list[k] = aligned_list[i - 1]
  #         # 如果后面存在值
  #         else:   
  #           # 如果连续缺失值超过一个 对i+1到i+continue_count-1的值赋予相同的值
  #           for k in range(i, i + continue_count - 1):
  #             aligned_list[k] = aligned_list[i - 1]
  #           before = aligned_list[i + continue_count - 2]
  #           after = aligned_list[i + continue_count]
  #           aligned_list[i + continue_count - 1] = (before + after) / 2
  #       i = i + continue_count
  #     else:
  #       i = i + 1
  #   # 如果最后一帧为空缺值
  #   last_frame = len(aligned_list) -1
  #   if aligned_list[last_frame] == []:
  #     aligned_list[last_frame] = aligned_list[last_frame - 1]
  #   return aligned_list
    
  # 人脸检测点缺失值填充
  def fill_null_coord(self, coordinate_list, video_name, null_save_path):
    i = 1
    continue_count_list = []
    # 从第2个找到倒数第二个
    while (i < len(coordinate_list) - 1):
      # 找到第一个不为缺失值的表格
      if coordinate_list.loc[i, 'coordinate_0'] == 'nan,nan':
        # 以i为起点继续向后找连续缺失值
        continue_count = 1
        for null_index in range(i + 1, len(coordinate_list)):
          if coordinate_list.loc[null_index, 'coordinate_0'] == 'nan,nan':
            continue_count += 1
          else:
            continue_count_list.append(continue_count)
            break
        # 填充缺失值：如果只有一个缺失值
        if continue_count == 1:
          for point in range(0,106):
            x_before, y_before = coordinate_list.loc[i - 1, 'coordinate_' + str(point)].split(',')
            x_after, y_after = coordinate_list.loc[i + 1, 'coordinate_' + str(point)].split(',')
            coordinate_list.loc[i, 'coordinate_' + str(point)] = str(round((int(x_before) + int(x_after)) / 2)) + ',' + \
                                                        str(round((int(y_before) + int(y_after)) / 2))
        if continue_count > 1:
          # 如果后面全是空缺值
          if i + continue_count -1 == len(coordinate_list) -1 :
            for k in range(i, i + continue_count - 1):
              coordinate_list.loc[k] = coordinate_list.loc[i - 1]
          # 如果后面存在值
          else:   
            # 如果连续缺失值超过一个 对i+1到i+continue_count-1的值赋予相同的值
            for k in range(i, i + continue_count - 1):
              coordinate_list.loc[k] = coordinate_list.loc[i - 1]
            for point in range(0, 106):
              x_before, y_before = coordinate_list.loc[i + continue_count - 2, 'coordinate_' + str(point)].split(',')
              x_after, y_after = coordinate_list.loc[i + continue_count, 'coordinate_' + str(point)].split(',')
              coordinate_list.loc[i + continue_count - 1, 'coordinate_' + str(point)] = str(round((int(x_before) + int(x_after)) / 2)) + ',' + \
                                                          str(round((int(y_before) + int(y_after)) / 2))
        i = i + continue_count
      else:
        i = i + 1
    # 如果最后一帧为空缺值
    last_frame = len(coordinate_list) -1
    if coordinate_list.loc[last_frame, 'coordinate_0'] == 'nan,nan':
        coordinate_list.loc[last_frame] = coordinate_list.loc[last_frame - 1]

    if continue_count_list != [] and max(continue_count_list) > 6:
      with open(null_save_path, 'a+') as f:
        f.write(video_name + ": " + str(continue_count_list))
        f.write('\n')
    return coordinate_list
