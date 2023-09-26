from math import sqrt
import numpy as np
import cv2


def group_salient_regions(input_list,frame_number,salient_scores,saliency_path,spatial_distance,fill_loss,resolution=[1024,2048]):
    # Initialize a dictionary to store the groups based on the x-axis value (first element of each tuple)
    groups_dict = {}
    groups_frame = {}

    groups_scores = {}




    # Iterate through the input list and group elements based on x-axis proximity
    for i,sublist in enumerate(input_list):
        for j,item in enumerate(sublist):
            #print("new item",item)
            added_to_group = False
            lower_distance = 10000
            lower_distance1 = 10000
            b = 10000
            group = 0

            for group_key in groups_dict.keys():
                #print(groups_dict[group_key])
                # Check if the current item can be grouped with the items in the group based on x-axis,y-axis proximity


                last_group_items = groups_dict[group_key][-1:]
                #print("last_group_items",last_group_items)


                for group_item in last_group_items:

                    a = sqrt((1.6*((item[0]+item[2]/2) - (group_item[0]+group_item[2]/2))**2) + (0.4*((item[1]+item[3]/2)-(group_item[1]+group_item[3]/2))**2)) # give a weight factor for x-axis and y-axis

                    if item[0]-200<0 or resolution[1]-item[0]+item[2]<100: #check if the salient regions are close from left side and right side of the ERP image
                        b = sqrt((1.6*(resolution[1]-max((item[0]+item[2]/2),(group_item[0]+group_item[2]/2))-min((item[0]+item[2]/2),(group_item[0]+group_item[2]/2)))**2) + (0.4*((item[1]+item[3]/2)-(group_item[1]+group_item[3]/2))**2))

                    if a <spatial_distance:
                        lower_distance = a
                        group = group_key
                        break


                    if b<spatial_distance:
                        lower_distance1 = b
                        group1 = group_key
                        break


                if lower_distance!=10000 or lower_distance1!=10000:

                    break


            if lower_distance<lower_distance1:

                if lower_distance!=10000 and lower_distance<spatial_distance:

                    groups_dict[group].append(item)
                    groups_frame[group].append(frame_number[i])
                    groups_scores[group].append(salient_scores[i][j])

                    added_to_group = True
            if lower_distance1<lower_distance:

                if lower_distance1 != 10000 and lower_distance1 <spatial_distance:

                    groups_dict[group1].append(item)
                    groups_frame[group1].append(frame_number[i])
                    groups_scores[group1].append(salient_scores[i][j])

                    added_to_group = True
            if not added_to_group:
                #print("no close, so new item",item)
                groups_dict[tuple(item)] = [item]
                groups_frame[tuple(item)]= [frame_number[i]]
                groups_scores[tuple(item)] = [salient_scores[i][j]]


            #Hold together the salient regions, the frames, the scores a
            groups_dict = dict(sorted(groups_dict.items(), key=lambda item: len(item[1]), reverse=True))
            groups_frame = dict(sorted(groups_frame.items(), key=lambda item: len(item[1]), reverse=True))
            groups_scores = dict(sorted(groups_scores.items(), key=lambda item: len(item[1]), reverse=True))

            #print("sorted_groupd_dict",groups_dict)

            #print(groups_dict)
            #groups_dict = dict(sorted(groups_dict.items(), key=lambda item: item[1]))
            #groups_frame = dict(sorted(groups_frame.items(), key=lambda item: item[1]))
    # Convert the dictionary values to the final lists
    result_lists = list(groups_dict.values())
    results_frames = list(groups_frame.values())
    results_scores = list(groups_scores.values())


    list_of_2d_volumes,list_of_2d_frames,lists_of_scores = fill_loss_frames(result_lists,results_frames,results_scores,fill_loss,saliency_path)


    return list_of_2d_volumes,list_of_2d_frames,lists_of_scores


def fill_loss_frames(lists_of_results1,lists_frames1,lists_of_scores1,num_frames,path):
    lists_of_results = []
    lists_frames = []
    lists_of_scores = []
    #print("lists of regions",lists_of_regions)
    lists_of_results1 = [sublist for sublist in lists_of_results1 if len(sublist) >= 150]
    lists_frames1= [sublist for sublist in lists_frames1 if len(sublist) >= 150]
    lists_of_scores1 = [sublist for sublist in lists_of_scores1 if len(sublist) >=150]


    for i in range(len(lists_of_results1)):
        new_frames = []
        new_results = []
        new_scores = []
        new_results1 = []
        flag = False
        for j in range(len(lists_frames1[i]) - 1):
            dif = int(lists_frames1[i][j + 1]) - int(lists_frames1[i][j])

            new_frames.append(int(lists_frames1[i][j]))
            new_results.append(lists_of_results1[i][j])
            new_scores.append(lists_of_scores1[i][j])
            if (dif) <= num_frames:

                for r in range(dif - 1):
                    x = int(lists_frames1[i][j]) + r + 1
                    new_frames.append(x)
                    new_results.append(lists_of_results1[i][j])


                    new_scores.append(0.05)
            else:
                lists_of_results.append(new_results)
                lists_frames.append(new_frames)
                lists_of_scores.append(new_scores)
                new_frames = []
                new_results = []
                new_scores = []

        lists_of_results.append(new_results)
        lists_frames.append(new_frames)
        lists_of_scores.append(new_scores)
    final_salient_regions = [sublist for sublist in lists_of_results if len(sublist) >= 150]
    final_frames = [sublist for sublist in lists_frames if len(sublist) >= 150]
    final_scores = [sublist for sublist in lists_of_scores if len(sublist)>= 150]



    combined = list(zip(final_frames, final_salient_regions,final_scores))
    combined.sort(key=lambda x: x[0])

    # Separate back into two lists
    final_frames, final_salient_regions,final_scores = zip(*combined)


    print(len(final_frames))
    return final_salient_regions,final_frames,final_scores