def single_path_finder_deprecated(signal, org_signal=None):
    current_val = 0
    start_index = 0
    end_index = 0
    max_prev = {'start': 0, 'end': 0, 'intensity': 0, 'velocity': 0}

    Oposite_threshold = 1
    Direction_last_few = 0  #
    Last_opposite_index = 0
    result_list = []
    current_direction = 0
    while (end_index < len(signal)):
        # print(
        #     f"{end_index}  ({org_signal[end_index]}, {signal[end_index]})  ---  Current_val: {current_val} -- Direction_last_few: {Direction_last_few} Last_opposite:{Last_opposite_index}\n")
        if current_direction == 0:

            if signal[end_index] == 0:
                start_index = end_index

            elif signal[end_index] > 0:  # signal[end_index-1]: # Increasing

                Direction_last_few += 1
                if abs(Direction_last_few) <= Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few 0+")
                    end_index += 1  # while step
                    continue

                end_index = Last_opposite_index - 1

                # State Change
                # Update value
                current_val += 1
                current_direction = +1
                # Store max_prev
                # Here we don't
                # result_list.append(max_prev)
                # Jump Stationary moments
                start_index = end_index
                intensity = current_val
                New_velocity = current_val / (end_index - start_index + 1)
                max_prev = {'start': start_index, 'end': end_index, 'intensity': intensity, 'velocity': New_velocity}
                Direction_last_few = +1

            elif signal[end_index] < 0:  # signal[end_index-1]: # Decreasing order

                Direction_last_few -= 1
                if abs(Direction_last_few) <= Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few0-")
                    end_index += 1  # while step
                    continue

                end_index = Last_opposite_index - 1
                # Direction_last_few is already modifed and can be used later in two other situation

                # State Change
                # Update value
                current_val += -1
                current_direction = - 1
                # Skip stationary movements
                start_index = end_index

                intensity = current_val
                New_velocity = current_val / (end_index - start_index + 1)
                max_prev = {'start': start_index, 'end': end_index, 'intensity': intensity, 'velocity': New_velocity}
                Direction_last_few = -1


        elif current_direction > 0:  # Increasing order

            if signal[end_index] == 0:
                pass

            elif signal[end_index] > 0:  # signal[end_index - 1]:  # Increasing
                # No Change
                # Update value
                current_val += 1

                Direction_last_few = Direction_last_few + 1 if (
                                                                   Direction_last_few) < Oposite_threshold else Direction_last_few

                # New_Worth = current_val / (end_index - start_index + 1)
                # if abs(max_prev['worth']) <= abs(New_Worth):
                #     max_prev = {'satrt': start_index, 'end': end_index, 'worth': New_Worth}
                Current_intensity = current_val
                Current_velocity = current_val / (end_index - start_index + 1)
                Current_velocity = round(Current_velocity, 2)
                if (max_prev['intensity']) < (Current_intensity):
                    max_prev = {'start': start_index, 'end': end_index, 'intensity': Current_intensity,
                                'velocity': Current_velocity}



            elif signal[end_index] < 0:  # signal[end_index - 1]:  # Decreasing order

                # current_val += -1   #add current direction
                Direction_last_few -= 1

                if (Direction_last_few) > -Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few-")
                    end_index += 1  # while step
                    current_val -= 1
                    continue

                # If we get to here it means that we already passed the threshold of opposite motion direction
                # We will back to the last time we observed the opposition which could be from "threshold" steps behind
                # to larger.
                # We set that as our new index to process the rest of the signal later (after this step) from that point
                # The result from considering Last_opposite_index to the current max_prev would definetly not appear, so no need to check.
                end_index = Last_opposite_index - 1  # because we also have stationarry current_Val = 0 at the begining of our loop?


                # State Change
                # Update value from start_index ---> end_index-1
                end_before_current_move = end_index - 1
                while start_index < end_index - 1:  # up to start the new direction
                    if signal[start_index] < 0:  # else = signal_start_index] = 0
                        current_val -= -1
                    if signal[start_index] > 0:  # else = signal_start_index] = 0
                        current_val -= 1
                    # New_Worth = current_val / ( (end_index-1) - start_index )
                    #
                    # if abs(max_prev['worth']) <= abs(New_Worth):
                    #     max_prev = {'satrt': start_index + 1, 'end': end_index-1, 'worth': New_Worth}

                    Current_intensity = current_val
                    Current_velocity = current_val / ((end_index - 1) - start_index)
                    if ((max_prev['intensity']) < (Current_intensity)) or \
                            ((max_prev['intensity']) == (Current_intensity) and abs(max_prev['velocity']) < abs(
                                Current_velocity)):
                        max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}

                    start_index += 1

                result_list.append(max_prev)
                start_index = end_index
                max_prev = {'start': start_index, 'end': end_index, 'intensity': -1, 'velocity': -1}  # opposite val

                current_val = 0  # -1
                Direction_last_few = -1
                current_direction = 0


        # Decreasing flow
        elif current_direction < 0:  # todo: fix the following

            if signal[end_index] == 0:
                pass

            elif signal[end_index] > 0:  # signal[end_index - 1]:  # Increasing Thanks to '<', '>' we also handle big jumps in rotations

                # current_val += +1
                Direction_last_few += 1
                if (Direction_last_few) < Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few-")
                    end_index += 1  # while step
                    current_val += 1
                    continue
                end_index = Last_opposite_index - 1


                # State Change
                # Update value

                # Todo: here we should loop start_index to end_index to consider the other side of the path

                while start_index < end_index - 1:
                    if signal[start_index] < 0:  # else = signal_start_index] = 0
                        current_val -= -1
                    if signal[start_index] > 0:  # else = signal_start_index] = 0
                        current_val -= 1

                    # Update max prev
                    # New_Worth = current_val / ( (end_index-1) - start_index) #Exclusive start
                    # if abs(max_prev['worth']) <= abs(New_Worth): # for 1/1, 2/2, ...  We should also consider cval comparison
                    #     max_prev = {'satrt': start_index + 1, 'end': end_index-1, 'worth': New_Worth}
                    Current_intensity = current_val
                    Current_velocity = current_val / ((end_index - 1) - start_index)
                    if (abs(max_prev['intensity']) < (Current_intensity)) or \
                            ((max_prev['intensity']) == (Current_intensity) and
                             abs(max_prev['velocity']) < abs(Current_velocity)):
                        max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}

                    start_index += 1

                result_list.append(max_prev)
                start_index = end_index
                max_prev = {'start': start_index, 'end': end_index, 'intensity': +1, 'velocity': 1}  # opposite val

                current_val = 0  # -1
                Direction_last_few = +1
                current_direction = 0

            elif signal[end_index] < 0:  # signal[end_index - 1]:  # Decreasing

                Direction_last_few = Direction_last_few - 1 if abs(
                    Direction_last_few) < Oposite_threshold else Direction_last_few
                # No Change
                # Update value
                current_val += -1
                # New_Worth = current_val / (end_index - start_index + 1)
                # if abs(New_Worth) >= abs(max_prev['worth']):
                #     max_prev = {'satrt': start_index, 'end': end_index, 'worth': New_Worth}

                Current_intensity = current_val
                Current_velocity = current_val / (end_index - start_index + 1)
                Current_velocity = round(Current_velocity, 2)
                if abs(max_prev['intensity']) < abs(
                        Current_intensity):  # it would be always the case unless we add more conditions
                    max_prev = {'start': start_index, 'end': end_index, 'intensity': Current_intensity,
                                'velocity': Current_velocity}

        # ---------------STEP---------------
        end_index += 1
    # Tod: ending

    if start_index < end_index:
        if current_val > 0:
            while start_index < end_index - 1:  # up to start the new direction
                if signal[start_index] < 0:  # else = signal_start_index] = 0
                    current_val -= -1
                if signal[start_index] > 0:  # else = signal_start_index] = 0
                    current_val -= 1
                Current_intensity = current_val
                Current_velocity = current_val / ((end_index - 1) - start_index)
                Current_velocity = round(Current_velocity, 2)
                if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                        (abs(max_prev['intensity']) == abs(Current_intensity) and abs(max_prev['velocity']) < abs(
                            Current_velocity)):
                    max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                'velocity': Current_velocity}
                start_index += 1

        if current_val < 0:
            while start_index < end_index - 1:
                if signal[start_index] < 0:  # else = signal_start_index] = 0
                    current_val -= -1
                if signal[start_index] > 0:  # else = signal_start_index] = 0
                    current_val -= 1
                Current_intensity = current_val
                Current_velocity = current_val / ((end_index - 1) - start_index)
                Current_velocity = round(Current_velocity, 2)
                if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                        (abs(max_prev['intensity']) == abs(Current_intensity) and
                         abs(max_prev['velocity']) < abs(Current_velocity)):
                    max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                'velocity': Current_velocity}

                start_index += 1
        if abs(max_prev['intensity'])>0:
            result_list.append(max_prev)

    # print('Final Result \n')
    # # print('*' * 20)
    # print(f'org_Signal: \n{", ".join(map(str, org_signal))}\n')
    # print(f'ResultS:\n {result_list}')
    # print('*' * 20)

    return result_list


def single_limitted_path_finder(signal, org_signal=None, range=5):
    current_val = 0
    start_index = 0
    end_index = 0
    max_prev = {'start': 0, 'end': 0, 'intensity': 0, 'velocity': 0}

    Oposite_threshold = 1
    Direction_last_few = 0  #
    Last_opposite_index = 0
    result_list = []
    current_direction = 0
    while (end_index < len(signal)):
        if abs(max_prev['velocity'])>1:
            print("!!!!!!!!!")
        print(
            f"{end_index}  ({org_signal[end_index]}, {signal[end_index]})  --- CD: {current_direction} Current_val: {current_val} -- Direction_last_few: {Direction_last_few} Last_opposite:{Last_opposite_index}\n")

        # while end_index-start_index > range:
        #
        #     if current_val == 0:
        #         if signal[start_index] == 0:
        #             continue
        #         elif signal[start_index] > 0:
        #             pass
        #         elif signal[start_index] < 0:
        #             pass
        #
        #     if current_val > 0:
        #         pass
        #     if current_val<0:
        #         pass
        #
        #     current_direction -= signal[start_index]
        #     current_val -= signal[start_index]
        #
        #     Direction_last_few = Direction_last_few + signal[start_index] if abs(
        #         Direction_last_few) < Oposite_threshold else Direction_last_few
        #     start_index += 1

        if current_direction == 0:

            if signal[end_index] == 0:
                start_index = end_index # todo: it might be set to the last_opposite

            elif signal[end_index] > 0:  # signal[end_index-1]: # Increasing

                Direction_last_few += 1
                if abs(Direction_last_few) <= Oposite_threshold:
                    # Todo: check if it is actually in the opposite direction
                    if (signal[end_index] > 0 and signal[Last_opposite_index] < 0) or \
                            signal[end_index] < 0 and signal[Last_opposite_index] > 0:
                        Last_opposite_index = end_index

                    # print("Direction_last_few 0+")
                    end_index += 1  # while step
                    continue

                end_index = Last_opposite_index  # -1 # Todo: I think this should be without -1

                # State Change
                # Update value
                current_val += 0
                current_direction = +1
                # Store max_prev
                # Here we don't
                # result_list.append(max_prev)
                # Jump Stationary moments
                start_index = end_index
                intensity = current_val
                New_velocity = current_val / (end_index - start_index + 1)
                max_prev = {'start': start_index, 'end': end_index, 'intensity': intensity, 'velocity': New_velocity}
                Direction_last_few = +1

            elif signal[end_index] < 0:  # signal[end_index-1]: # Decreasing order

                Direction_last_few -= 1
                if abs(Direction_last_few) <= Oposite_threshold:
                    # Todo: check if it is actually in the opposite direction
                    if (signal[end_index]>0 and signal[Last_opposite_index] < 0) or\
                        signal[end_index]<0 and signal[Last_opposite_index] > 0:
                        Last_opposite_index = end_index

                    # print("Direction_last_few0-")
                    end_index += 1  # while step
                    continue

                end_index = Last_opposite_index  #-1 # Todo: I think this should be without -1
                # Direction_last_few is already modifed and can be used later in two other situation

                # State Change
                # Update value
                current_val += 0
                current_direction = - 1
                # Skip stationary movements
                start_index = end_index

                intensity = current_val
                New_velocity = current_val / (end_index - start_index + 1)
                max_prev = {'start': start_index, 'end': end_index, 'intensity': intensity, 'velocity': New_velocity}
                Direction_last_few = -1


        elif current_direction > 0:  # Increasing order

            if signal[end_index] == 0:
                pass

            elif signal[end_index] > 0:  # signal[end_index - 1]:  # Increasing
                # No Change
                # Update value
                current_val += 1

                Direction_last_few = Direction_last_few + 1 if (
                                                                   Direction_last_few) < Oposite_threshold else Direction_last_few

                # New_Worth = current_val / (end_index - start_index + 1)
                # if abs(max_prev['worth']) <= abs(New_Worth):
                #     max_prev = {'satrt': start_index, 'end': end_index, 'worth': New_Worth}
                Current_intensity = current_val
                Current_velocity = current_val / (end_index - start_index + 1)
                Current_velocity = round(Current_velocity, 2)

                duration = (end_index - start_index + 1)
                if duration <= range:
                    if (max_prev['intensity']) < (Current_intensity):
                        max_prev = {'start': start_index, 'end': end_index, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}



            elif signal[end_index] < 0:  # signal[end_index - 1]:  # Decreasing order

                # current_val += -1   #add current direction
                Direction_last_few -= 1

                if (Direction_last_few) > -Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few-")
                    end_index += 1  # while step
                    current_val -= 1
                    continue

                # If we get to here it means that we already passed the threshold of opposite motion direction
                # We will back to the last time we observed the opposition which could be from "threshold" steps behind
                # to larger.
                # We set that as our new index to process the rest of the signal later (after this step) from that point
                # The result from considering Last_opposite_index to the current max_prev would definetly not appear, so no need to check.
                end_index = Last_opposite_index - 1  # because we also have stationarry current_Val = 0 at the begining of our loop?


                # State Change
                # Update value from start_index ---> end_index-1
                end_before_current_move = end_index - 1
                while start_index < end_index - 1:  # up to start the new direction
                    if signal[start_index] < 0:  # else = signal_start_index] = 0
                        current_val -= -1
                    if signal[start_index] > 0:  # else = signal_start_index] = 0
                        current_val -= 1
                    # New_Worth = current_val / ( (end_index-1) - start_index )
                    #
                    # if abs(max_prev['worth']) <= abs(New_Worth):
                    #     max_prev = {'satrt': start_index + 1, 'end': end_index-1, 'worth': New_Worth}

                    Current_intensity = current_val
                    Current_velocity = current_val / ((end_index - 1) - start_index)
                    duration = (end_index - start_index + 1)
                    if duration<=range:
                        if ((max_prev['intensity']) < (Current_intensity)) or \
                                ((max_prev['intensity']) == (Current_intensity) and abs(max_prev['velocity']) < abs(
                                    Current_velocity)):
                            max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                        'velocity': Current_velocity}

                    start_index += 1

                result_list.append(max_prev)
                start_index = end_index
                max_prev = {'start': start_index, 'end': end_index, 'intensity': -1, 'velocity': -1}  # opposite val

                current_val = 0  # -1
                Direction_last_few = -1
                current_direction = 0


        # Decreasing flow
        elif current_direction < 0:  # todo: fix the following

            if signal[end_index] == 0:
                pass

            elif signal[end_index] > 0:  # signal[end_index - 1]:  # Increasing Thanks to '<', '>' we also handle big jumps in rotations

                # current_val += +1
                Direction_last_few += 1
                if (Direction_last_few) < Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few-")
                    end_index += 1  # while step
                    current_val += 1
                    continue
                end_index = Last_opposite_index - 1
                # current_val += -1 # to undo the prev. +1 at the last opposit that has been seen

                # State Change
                # Update value

                # Todo: here we should loop start_index to end_index to consider the other side of the path

                while start_index < end_index - 1:
                    if signal[start_index] < 0:  # else = signal_start_index] = 0
                        current_val -= -1
                    if signal[start_index] > 0:  # else = signal_start_index] = 0
                        current_val -= 1

                    # Update max prev
                    # New_Worth = current_val / ( (end_index-1) - start_index) #Exclusive start
                    # if abs(max_prev['worth']) <= abs(New_Worth): # for 1/1, 2/2, ...  We should also consider cval comparison
                    #     max_prev = {'satrt': start_index + 1, 'end': end_index-1, 'worth': New_Worth}
                    Current_intensity = current_val
                    Current_velocity = current_val / ((end_index - 1) - start_index)

                    duration = (end_index - start_index + 1)
                    if duration <= range:
                        if (abs(max_prev['intensity']) < (Current_intensity)) or \
                                ((max_prev['intensity']) == (Current_intensity) and
                                 abs(max_prev['velocity']) < abs(Current_velocity)):
                            max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                        'velocity': Current_velocity}

                    start_index += 1
                if abs(max_prev['velocity'])>1:
                    print("!!")

                result_list.append(max_prev)
                start_index = end_index
                max_prev = {'start': start_index, 'end': end_index, 'intensity': +1, 'velocity': 1}  # opposite val

                current_val = 0  # -1
                Direction_last_few = +1
                current_direction = 0

            elif signal[end_index] < 0:  # signal[end_index - 1]:  # Decreasing

                Direction_last_few = Direction_last_few - 1 if abs(
                    Direction_last_few) < Oposite_threshold else Direction_last_few
                # No Change
                # Update value
                current_val += -1
                # New_Worth = current_val / (end_index - start_index + 1)
                # if abs(New_Worth) >= abs(max_prev['worth']):
                #     max_prev = {'satrt': start_index, 'end': end_index, 'worth': New_Worth}

                Current_intensity = current_val
                Current_velocity = current_val / (end_index - start_index + 1)
                Current_velocity = round(Current_velocity, 2)

                duration = (end_index - start_index + 1)
                if duration <= range:
                    if abs(max_prev['intensity']) < abs(
                            Current_intensity):  # it would be always the case unless we add more conditions
                        max_prev = {'start': start_index, 'end': end_index, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}

        # ---------------STEP---------------
        end_index += 1
    # Todo: ending

    if start_index < end_index:
        if current_val > 0:
            while start_index < end_index - 1:  # up to start the new direction
                if signal[start_index] < 0:  # else = signal_start_index] = 0
                    current_val -= -1
                if signal[start_index] > 0:  # else = signal_start_index] = 0
                    current_val -= 1
                Current_intensity = current_val
                Current_velocity = current_val / ((end_index - 1) - start_index)
                Current_velocity = round(Current_velocity, 2)

                duration = (end_index - start_index + 1)
                if duration <= range:
                    if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                            (abs(max_prev['intensity']) == abs(Current_intensity) and abs(max_prev['velocity']) < abs(
                                Current_velocity)):
                        max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}
                start_index += 1

        if current_val < 0:
            while start_index < end_index - 1:
                if signal[start_index] < 0:  # else = signal_start_index] = 0
                    current_val -= -1
                if signal[start_index] > 0:  # else = signal_start_index] = 0
                    current_val -= 1
                Current_intensity = current_val
                Current_velocity = current_val / ((end_index - 1) - start_index)
                Current_velocity = round(Current_velocity, 2)

                duration = (end_index - start_index + 1)
                if duration <= range:

                    if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                            (abs(max_prev['intensity']) == abs(Current_intensity) and
                             abs(max_prev['velocity']) < abs(Current_velocity)):
                        max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}

                start_index += 1
        if abs(max_prev['intensity'])>0:
            result_list.append(max_prev)

    # print('Final Result \n')
    # # print('*' * 20)
    # print(f'org_Signal: \n{", ".join(map(str, org_signal))}\n')
    # print(f'ResultS:\n {result_list}')
    # print('*' * 20)

    return result_list



def OLD_single_limitted_path_finder(signal, org_signal=None, range=5):
    current_val = 0
    start_index = 0
    end_index = 0
    max_prev = {'start': 0, 'end': 0, 'intensity': 0, 'velocity': 0}

    Oposite_threshold = 1
    Direction_last_few = 0  #
    Last_opposite_index = 0
    result_list = []
    current_direction = 0
    while (end_index < len(signal)):
        print(
            f"{end_index}  ({org_signal[end_index]}, {signal[end_index]})  --- CD:{current_direction}  Current_val: {current_val} -- Direction_last_few: {Direction_last_few} Last_opposite:{Last_opposite_index}\n")

        # while end_index-start_index > range:
        #
        #     if current_val == 0:
        #         if signal[start_index] == 0:
        #             continue
        #         elif signal[start_index] > 0:
        #             pass
        #         elif signal[start_index] < 0:
        #             pass
        #
        #     if current_val > 0:
        #         pass
        #     if current_val<0:
        #         pass
        #
        #     current_direction -= signal[start_index]
        #     current_val -= signal[start_index]
        #
        #     Direction_last_few = Direction_last_few + signal[start_index] if abs(
        #         Direction_last_few) < Oposite_threshold else Direction_last_few
        #     start_index += 1

        if current_direction == 0:

            if signal[end_index] == 0:
                start_index = end_index

            elif signal[end_index] > 0:  # signal[end_index-1]: # Increasing

                Direction_last_few += 1
                if abs(Direction_last_few) <= Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few 0+")
                    end_index += 1  # while step
                    continue

                end_index = Last_opposite_index - 1

                # State Change
                # Update value
                current_val += 1
                current_direction = +1
                # Store max_prev
                # Here we don't
                # result_list.append(max_prev)
                # Jump Stationary moments
                start_index = end_index
                intensity = current_val
                New_velocity = current_val / (end_index - start_index + 1)
                max_prev = {'start': start_index, 'end': end_index, 'intensity': intensity, 'velocity': New_velocity}
                Direction_last_few = +1

            elif signal[end_index] < 0:  # signal[end_index-1]: # Decreasing order

                Direction_last_few -= 1
                if abs(Direction_last_few) <= Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few0-")
                    end_index += 1  # while step
                    continue

                end_index = Last_opposite_index - 1
                # Direction_last_few is already modifed and can be used later in two other situation

                # State Change
                # Update value
                current_val += -1
                current_direction = - 1
                # Skip stationary movements
                start_index = end_index

                intensity = current_val
                New_velocity = current_val / (end_index - start_index + 1)
                max_prev = {'start': start_index, 'end': end_index, 'intensity': intensity, 'velocity': New_velocity}
                Direction_last_few = -1


        elif current_direction > 0:  # Increasing order

            if signal[end_index] == 0:
                pass

            elif signal[end_index] > 0:  # signal[end_index - 1]:  # Increasing
                # No Change
                # Update value
                current_val += 1

                Direction_last_few = Direction_last_few + 1 if (
                                                                   Direction_last_few) < Oposite_threshold else Direction_last_few

                # New_Worth = current_val / (end_index - start_index + 1)
                # if abs(max_prev['worth']) <= abs(New_Worth):
                #     max_prev = {'satrt': start_index, 'end': end_index, 'worth': New_Worth}
                Current_intensity = current_val
                Current_velocity = current_val / (end_index - start_index + 1)
                Current_velocity = round(Current_velocity, 2)

                duration = (end_index - start_index + 1)
                if duration <= range:
                    if (max_prev['intensity']) < (Current_intensity):
                        max_prev = {'start': start_index, 'end': end_index, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}



            elif signal[end_index] < 0:  # signal[end_index - 1]:  # Decreasing order

                # current_val += -1   #add current direction
                Direction_last_few -= 1

                if (Direction_last_few) > -Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few-")
                    end_index += 1  # while step
                    current_val -= 1
                    continue

                # If we get to here it means that we already passed the threshold of opposite motion direction
                # We will back to the last time we observed the opposition which could be from "threshold" steps behind
                # to larger.
                # We set that as our new index to process the rest of the signal later (after this step) from that point
                # The result from considering Last_opposite_index to the current max_prev would definetly not appear, so no need to check.
                end_index = Last_opposite_index - 1  # because we also have stationarry current_Val = 0 at the begining of our loop?


                # State Change
                # Update value from start_index ---> end_index-1
                end_before_current_move = end_index - 1
                while start_index < end_index - 1:  # up to start the new direction
                    if signal[start_index] < 0:  # else = signal_start_index] = 0
                        current_val -= -1
                    if signal[start_index] > 0:  # else = signal_start_index] = 0
                        current_val -= 1
                    # New_Worth = current_val / ( (end_index-1) - start_index )
                    #
                    # if abs(max_prev['worth']) <= abs(New_Worth):
                    #     max_prev = {'satrt': start_index + 1, 'end': end_index-1, 'worth': New_Worth}

                    Current_intensity = current_val
                    Current_velocity = current_val / ((end_index - 1) - start_index)
                    duration = (end_index - start_index + 1)
                    if duration<=range:
                        if ((max_prev['intensity']) < (Current_intensity)) or \
                                ((max_prev['intensity']) == (Current_intensity) and abs(max_prev['velocity']) < abs(
                                    Current_velocity)):
                            max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                        'velocity': Current_velocity}

                    start_index += 1

                result_list.append(max_prev)
                start_index = end_index
                max_prev = {'start': start_index, 'end': end_index, 'intensity': -1, 'velocity': -1}  # opposite val

                current_val = 0  # -1
                Direction_last_few = -1
                current_direction = 0


        # Decreasing flow
        elif current_direction < 0:  # todo: fix the following

            if signal[end_index] == 0:
                pass

            elif signal[end_index] > 0:  # signal[end_index - 1]:  # Increasing Thanks to '<', '>' we also handle big jumps in rotations

                # current_val += +1
                Direction_last_few += 1
                if (Direction_last_few) < Oposite_threshold:
                    Last_opposite_index = end_index
                    # print("Direction_last_few-")
                    end_index += 1  # while step
                    current_val += 1
                    continue
                end_index = Last_opposite_index - 1


                # State Change
                # Update value

                # Todo: here we should loop start_index to end_index to consider the other side of the path

                while start_index < end_index - 1:
                    if signal[start_index] < 0:  # else = signal_start_index] = 0
                        current_val -= -1
                    if signal[start_index] > 0:  # else = signal_start_index] = 0
                        current_val -= 1

                    # Update max prev
                    # New_Worth = current_val / ( (end_index-1) - start_index) #Exclusive start
                    # if abs(max_prev['worth']) <= abs(New_Worth): # for 1/1, 2/2, ...  We should also consider cval comparison
                    #     max_prev = {'satrt': start_index + 1, 'end': end_index-1, 'worth': New_Worth}
                    Current_intensity = current_val
                    Current_velocity = current_val / ((end_index - 1) - start_index)

                    duration = (end_index - start_index + 1)
                    if duration <= range:
                        if (abs(max_prev['intensity']) < (Current_intensity)) or \
                                ((max_prev['intensity']) == (Current_intensity) and
                                 abs(max_prev['velocity']) < abs(Current_velocity)):
                            max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                        'velocity': Current_velocity}

                    start_index += 1

                result_list.append(max_prev)
                start_index = end_index
                max_prev = {'start': start_index, 'end': end_index, 'intensity': +1, 'velocity': 1}  # opposite val

                current_val = 0  # -1
                Direction_last_few = +1
                current_direction = 0

            elif signal[end_index] < 0:  # signal[end_index - 1]:  # Decreasing

                Direction_last_few = Direction_last_few - 1 if abs(
                    Direction_last_few) < Oposite_threshold else Direction_last_few
                # No Change
                # Update value
                current_val += -1
                # New_Worth = current_val / (end_index - start_index + 1)
                # if abs(New_Worth) >= abs(max_prev['worth']):
                #     max_prev = {'satrt': start_index, 'end': end_index, 'worth': New_Worth}

                Current_intensity = current_val
                Current_velocity = current_val / (end_index - start_index + 1)
                Current_velocity = round(Current_velocity, 2)

                duration = (end_index - start_index + 1)
                if duration <= range:
                    if abs(max_prev['intensity']) < abs(
                            Current_intensity):  # it would be always the case unless we add more conditions
                        max_prev = {'start': start_index, 'end': end_index, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}

        # ---------------STEP---------------
        end_index += 1
    # Tod: ending

    if start_index < end_index:
        if current_val > 0:
            while start_index < end_index - 1:  # up to start the new direction
                if signal[start_index] < 0:  # else = signal_start_index] = 0
                    current_val -= -1
                if signal[start_index] > 0:  # else = signal_start_index] = 0
                    current_val -= 1
                Current_intensity = current_val
                Current_velocity = current_val / ((end_index - 1) - start_index)
                Current_velocity = round(Current_velocity, 2)

                duration = (end_index - start_index + 1)
                if duration <= range:
                    if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                            (abs(max_prev['intensity']) == abs(Current_intensity) and abs(max_prev['velocity']) < abs(
                                Current_velocity)):
                        max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}
                start_index += 1

        if current_val < 0:
            while start_index < end_index - 1:
                if signal[start_index] < 0:  # else = signal_start_index] = 0
                    current_val -= -1
                if signal[start_index] > 0:  # else = signal_start_index] = 0
                    current_val -= 1
                Current_intensity = current_val
                Current_velocity = current_val / ((end_index - 1) - start_index)
                Current_velocity = round(Current_velocity, 2)

                duration = (end_index - start_index + 1)
                if duration <= range:

                    if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                            (abs(max_prev['intensity']) == abs(Current_intensity) and
                             abs(max_prev['velocity']) < abs(Current_velocity)):
                        max_prev = {'start': start_index + 1, 'end': end_index - 1, 'intensity': Current_intensity,
                                    'velocity': Current_velocity}

                start_index += 1
        if abs(max_prev['intensity'])>0:
            result_list.append(max_prev)

    # print('Final Result \n')
    # # print('*' * 20)
    # print(f'org_Signal: \n{", ".join(map(str, org_signal))}\n')
    # print(f'ResultS:\n {result_list}')
    # print('*' * 20)

    return result_list



def single_path_finder(time_series_signal, threshold=1):

    # This is the latest algorithm I designed to extract a motion.
    # Designed, tested, and verified on periodic, and nor-periodic
    # test cases on 2023-10-11
    # We rename it from detect_movements_new to "single_path_finder"
    # for being used in the captioning.py

    # This adjustment enables the algorithm being robust against noises and rotary motions
    for i in range(len(time_series_signal)):
        if time_series_signal[i]>0: time_series_signal[i]=1
        if time_series_signal[i]<0: time_series_signal[i]=-1


    start_i, end_i = 0, 0
    last_few_neg = []
    last_few_pos = []
    Direction = 0


    positions = time_series_signal
    Final_output = []
    while start_i < len(positions)-1:

        start_i += 1

        diff = positions[start_i]

        if diff < 0:
            last_few_neg.append(start_i)
            Direction += -1 # diff # -1
        elif diff > 0:
            last_few_pos.append(start_i)
            Direction +=  1 # diff # 1




        if abs(Direction) > threshold:  # we detected a moton start from threshold steps ago
            # 1. Determine the start frame of the motion

            if Direction > 0:
                if len(last_few_pos) == 0:
                    pass
                elif len(last_few_pos) < threshold + 1:
                    start_i = last_few_pos[0]
                else:
                    start_i = last_few_pos[-(threshold+1)]
            if Direction < 0:
                if len(last_few_neg) == 0:
                    pass
                elif len(last_few_neg) < threshold + 1:
                    start_i = last_few_neg[0]
                else:
                    start_i = last_few_neg[-(threshold + 1)]

            max_prev = {'start': start_i,
                        'end': start_i,
                        'intensity': positions[start_i],
                        'velocity': 1}

            # 1.2: set the initial setting
            Current_intensity = positions[start_i]

            # 2. Determine the end of the segment (until we observe an opposite direction)


            inner_direction = 1 if Direction > 0 else -1
            end_i = start_i

            while end_i < len(positions)-1:

                end_i += 1
                # diff = 1 if positions[end_i] > 0 else 0
                # diff = -1 if positions[end_i] < 0 else diff
                diff = positions[end_i]
                if diff == 0: continue
                # Since we are going from left to right, we do trimming on the right
                #  side of the detected motion
                Current_intensity += diff  # we are adding this guy
                Current_velocity = Current_intensity / (end_i - start_i + 1)
                Current_velocity = round(Current_velocity, 2)

                if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                        (abs(max_prev['intensity']) == abs(Current_intensity) and abs(max_prev['velocity']) < abs(
                            Current_velocity)):

                    max_prev = {'start': start_i, 'end': end_i, 'intensity': Current_intensity,
                                'velocity': Current_velocity}

                if Direction > 0:
                    if diff > 0:
                        inner_direction = min(threshold, inner_direction + diff)
                    if diff < 0:
                        inner_direction = inner_direction + diff
                        if abs(inner_direction) >= threshold:
                            break
                if Direction < 0:
                    if diff > 0:
                        inner_direction = inner_direction + diff
                        if abs(inner_direction) >= threshold:
                            break
                    if diff < 0:
                        inner_direction = max(-threshold, inner_direction + diff)


            # 3. finding the minimum length with the maximum possible intensity (trim)
            Current_intensity = max_prev['intensity']
            end_i = max_prev['end']
            while start_i < end_i:
                start_i += 1
                # diff = 1 if positions[start_i] > 0 else 0
                # diff = -1 if positions[start_i] < 0 else 0
                diff = positions[start_i-1]
                # Since we are going from left to right, we do trimming on the right
                #  side of the detected motion
                Current_intensity -= diff  # we are removing it.
                Current_velocity = Current_intensity / (end_i - start_i + 1)
                Current_velocity = round(Current_velocity, 2)
                if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                        (abs(max_prev['intensity']) == abs(Current_intensity) and abs(max_prev['velocity']) < abs(
                            Current_velocity)):
                    max_prev = {'start': start_i, 'end': end_i, 'intensity': Current_intensity,
                                'velocity': Current_velocity}






            Final_output.append(max_prev)
            start_i = max_prev['end'] # --> later + 1
            Direction = 0
            last_few_neg = []
            last_few_pos = []

    # # Determine the maximum width needed for each column
    # start_width = max(len(str(item['start'])) for item in Final_output)
    # end_width = max(len(str(item['end'])) for item in Final_output)
    # intensity_width = max(len(str(item['intensity'])) for item in Final_output)
    # velocity_width = max(len(str(item['velocity'])) for item in Final_output)
    #
    # # Print headers
    # header_format = f"{{:>{start_width}}} | {{:>{end_width}}} | {{:>{intensity_width}}} | {{:>{velocity_width}}}"
    # print(header_format.format("Start", "End", "Intensity", "Velocity"))
    # print('-' * (start_width + end_width + intensity_width + velocity_width + 9))  # for "|", spaces and "-" characters
    #
    # # Print data
    # row_format = header_format
    # for item in Final_output:
    #     print(row_format.format(item['start'], item['end'], item['intensity'], item['velocity']))

    # for x in Final_output:
    #     print(x)
    return Final_output
    print(Final_output)
    return movements, movement_segments

# This function finds the minimum set

# This function finds the minimum aggregated posecodes with muinimum unwanted posecodes
# to cover the entire required pose set to cover. There is also a chance of finding a set
# that doesn't cover the entire required ones due to the unavailablle of uneligble posecodes
from copy import deepcopy
def min_samples_to_cover(required_pose_set, covered_pose_set):
    # Create a list of required pose sets not covered yet
    uncovered = set(required_pose_set)
    covered_pose_set_copy = deepcopy(covered_pose_set)
    # Create an empty list to store the selected samples
    selected_samples = []

    # Create an empty list to store the indices of selected samples
    selected_indices = []

    # While there are uncovered pose sets
    while uncovered and covered_pose_set:
        # Find the sample that covers the most uncovered pose sets
        best_sample = max(covered_pose_set, key=lambda sample: (len(sample.intersection(uncovered)), -len(sample)))
        print(best_sample)
        # Add the best sample to the selected samples
        if len(best_sample.intersection(uncovered)):
            selected_samples.append(best_sample)

            # Add the index of the best sample to the selected indices
            selected_indices.append(covered_pose_set_copy.index(best_sample))

            # Update the uncovered pose sets
            uncovered -= best_sample

        # Remove the best sample from the covered pose sets
        covered_pose_set.remove(best_sample)

    return selected_samples, selected_indices





# For experimental purposes
if False:

    time_series_signal =[49, 48, 49, 49, 49, 49, 49, 48, 49, 49, 49, 49, 49, 48, 48, 49, 48, 48, 49, 49, 49, 49, 49, 48, 48, 48, 49, 48, 48, 48, 49, 48, 49, 48, 48, 49, 48, 48, 48, 49, 48, 49, 48, 49, 49, 48, 48, 48, 48, 48, 49, 48, 48, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 47, 48, 47, 47, 47, 47, 47, 48, 48, 48, 48, 47, 48, 48, 48, 47, 47, 47, 48, 48, 47, 47, 48, 48, 48, 47, 47, 48, 48, 47, 48, 48, 48, 47, 47, 47, 47, 47, 48, 47, 48, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 46, 47, 47, 46, 47, 47, 47, 47, 46, 47, 47, 47, 47, 47, 46, 47, 47, 47, 46, 46, 47, 46, 46, 46, 46, 46, 47, 46, 46, 46, 46, 47, 46, 47, 46, 47, 46, 46, 47, 46, 46, 46, 46, 47, 46, 46, 47, 46, 46, 47, 46, 47, 46, 46, 46, 46, 47, 47, 46, 47, 46, 46, 47, 47, 46, 46, 46, 47, 46, 46, 46, 46, 46, 46, 47, 46, 46, 46, 47, 46, 46, 47, 46, 46, 46, 46, 46, 47, 46, 46, 47, 46, 46, 46, 46, 47, 47, 47, 46, 46, 47, 46, 47, 47, 47, 46, 47, 46, 46, 46, 46, 46, 47, 46, 47, 47, 47, 47, 46, 47, 47, 46, 47, 47, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 46, 47, 47, 47, 47, 46, 47, 46, 47, 46, 46, 46, 46, 47, 47

    ]
    time_series_signal = [2, 2, 2, 2, 3, 4, 5, 5, 4, 3, 2, 2, 2, 2, 2]

    time_series_signal = [130, 129, 128, 129, 128, 127, 128, 129, 129, 130, 128, 128, 129,
       128, 128, 128, 129, 129, 128, 129, 128, 128, 127, 129, 128, 129,
       128, 129, 128, 129, 129, 128, 129, 129, 128, 128, 127, 129, 129,
       128, 128, 128, 129, 128, 128, 129, 128, 127, 128, 128, 129, 127,
       128, 129, 127, 128, 129, 128, 128, 127, 128, 128, 129, 128, 127,
       129, 128, 128, 128, 129, 128, 129, 128, 128, 129, 128, 129, 127,
       127, 128, 129, 128, 127, 128, 127, 128, 127, 127, 127, 128, 128,
       129, 127, 129, 128, 127, 127, 127, 128, 128]

    # time_series_signal = time_series_signal[30:45]

    # perioidc signal
    time_series_signal = [0, 1, 2, 3, 4, 4, 5, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1]


    delta_signal = [time_series_signal[i + 1] - time_series_signal[i] for i in range(len(time_series_signal) - 1)]
    delta_signal = [0] + delta_signal # to initiate with 0


    print(time_series_signal)
    print(delta_signal)
    list_of_objects = single_path_finder(delta_signal, time_series_signal)

    # delta_signal = [ abs(x) for x in delta_signal]
    # list_of_objects = OLD_single_limitted_path_finder(delta_signal, time_series_signal, range=500)
    # list_of_objects = single_limitted_path_finder(delta_signal, time_series_signal, range=500)
    list_of_objects = single_path_finder(delta_signal, time_series_signal)
    list_of_objects = detect_movements_New(delta_signal, threshold=1)

    import matplotlib.pyplot as plt
    plt.plot(range(len(time_series_signal)), time_series_signal, label='Input', linestyle='--')
    for res in list_of_objects:
      # selected_elements = list(range(res['start'], res['end']+1))

      selected_elements = []
      clip = []

      for i in range(res['start'], res['end']+1):
        val = time_series_signal[i]
        if res['intensity']>0:
          val += 0.05
        else:
          val += -0.05
        selected_elements.append(i)
        clip.append(val)


      plt.plot(selected_elements,
               clip,
               label= '+' if res['intensity']>0 else '-',
               color= 'green' if res['intensity']>0 else 'red'

               )
    plt.title("Positive and Negative directions are with Green and Red color respectievely")
    plt.show()

import random

'''
def find_minimal_covering_subsequences(signal, range_length, target=1):
    minimal_covering_subsequences = []
    start_index, end_index = 0, -1

    current_counter = 0
    best_counter = 0
    result = []
    active_indices = []
    while start_index < len(signal):
        # print(start_index)
        if signal[start_index] != target:
            start_index += 1
            continue

        end_index = min(start_index + range_length, len(signal) - 1)
        for trim_end in range(end_index, start_index-1, -1):
            if signal[trim_end] == target:
                break
        subseq = signal[start_index:trim_end+1]

        new_list = []
        for a_i in range(start_index, end_index+1):
            if signal[a_i] == target:
                new_list.append(a_i)
        # print(subseq)

        result.append(subseq)
        active_indices.append(new_list)

        start_index = end_index + 1


    return result, active_indices


# Example usage:
nums = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
range_length = 3
print(nums)
result, ac = find_minimal_covering_subsequences(nums, range_length)
print(result)
print('\n', ac)
'''





# _______________________________________________________________________________
# _______________________________________________________________________________
# _______________________________________________________________________________
# _______________________________________________________________________________
# _______________________________________________________________________________
# _______________________________________________________________________________
# _______________________________________________________________________________
# _______________________________________________________________________________

import matplotlib.pyplot as plt





def visualize_movements(positions, movement_segments):
    plt.figure(figsize=(10, 5))
    plt.plot(positions, 'b-', label='Positions')

    for segment, movement in movement_segments:
        if movement == "moving right":
            color = 'g'
        elif movement == "moving left":
            color = 'r'
        plt.plot(range(segment[0], segment[1] + 1), positions[segment[0]:segment[1] + 1], color=color, linewidth=2)

    plt.title('Refined Movement Detection in Positions')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.show()


# Example Usage
# positions = [1, 2, 2, 2, 5, 6, 6, 6, 7, 6, 5, 4, 3, 3, 3, 4, 5, 6]
# movements, movement_segments = detect_movements(positions, threshold=2)
# visualize_movements(positions, zip(movement_segments, movements))

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------


# Example Usage
positions = [1, 2, 2, 2, 5, 6, 6, 6, 7, 6, 5, 4, 3, 3, 3, 4, 5, 6]
positions = [130, 129, 128, 129, 128, 127, 128, 129, 129, 130, 128, 128, 129,
             128, 128, 128, 129, 129, 128, 129, 128, 128, 127, 129, 128, 129,
             128, 129, 128, 129, 129, 128, 129, 129, 128, 128, 127, 129, 129,
             128, 128, 128, 129, 128, 128, 129, 128, 127, 128, 128, 129, 127,
             128, 129, 127, 128, 129, 128, 128, 127, 128, 128, 129, 128, 127,
             129, 128, 128, 128, 129, 128, 129, 128, 128, 129, 128, 129, 127,
             127, 128, 129, 128, 127, 128, 127, 128, 127, 127, 127, 128, 128,
             129, 127, 129, 128, 127, 127, 127, 128, 128]

positions =  [5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, ]

# movements, movement_segments = detect_movements(positions, threshold=1)
# visualize_movements(positions, zip(movement_segments, movements))



# --------------------------------------------------------------------------------------------

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import webcolors

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageColor


from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor


from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np


from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np


def create_timeline_image_with_blinking_good(motioncodes, frame_number, total_frames):
    width = 2000  # Width for higher resolution
    height = 1500  # Height for more tracks and higher resolution
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 70)  # Font size adjusted
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial.ttf is not available

    # Define colors for the segments
    colors = ["blue", "green", "orange", "purple", "pink", "cyan", "magenta"]

    # Calculate total duration for normalization
    total_duration = total_frames # max(segment['end'] for segment in motioncodes)

    # Organize motion codes by joint
    joint_segments = {}
    for segment in motioncodes:


        # Adjust the joint names
        segment['joint1'] = ' '.join(word.strip().capitalize() for word in segment['joint1'].strip("[]").
                                     replace("'", "").split(',')).replace('None', '')
        segment['joint2'] = ' '.join(word.strip().capitalize() for word in segment['joint2'].strip("[]").
                                     replace("'", "").split(',')).replace('None', '')
        joint = segment['joint1']
        if joint not in joint_segments:
            joint_segments[joint] = []
        joint_segments[joint].append(segment)

    # Calculate maximum width of joint names
    max_joint_name_width = max(
        draw.textbbox((0, 0), joint, font=font)[2] for joint in joint_segments.keys()) + 20
    timeline_start_x = max_joint_name_width + 20  # Start timeline after the header text column

    # Create a dictionary to store y positions and rows for each joint
    joint_y_positions = {}
    base_y = 80
    segment_height = 100  # Height of each segment adjusted for readability
    row_gap = 20  # Space between rows within a joint track
    track_gap = 20  # Space between tracks of different joints

    # Define multi-track info
    joint2motioncodes = {}
    for mc in motioncodes:
        if mc['joint1'] not in joint2motioncodes:
            joint2motioncodes[mc['joint1']] = []
        joint2motioncodes[mc['joint1']].append(mc)

    multi_track_info = {}
    for joint, j_motioncodes in joint2motioncodes.items():
        multi_track_info[joint] = []
        for mc in j_motioncodes:
            placed = False
            for track in multi_track_info[joint]:
                if all(mc['start'] >= existing_mc['end'] or mc['end'] <= existing_mc['start'] for existing_mc in
                       track['m_list']):
                    track['m_list'].append(mc)
                    placed = True
                    break
            if not placed:
                multi_track_info[joint].append({'track_number': len(multi_track_info[joint]), 'm_list': [mc]})

    # Determine y positions for each joint based on multi-track info
    max_y = base_y
    for joint, tracks in multi_track_info.items():
        joint_y_positions[joint] = {'base_y': base_y, 'rows': tracks}
        n_track = len(tracks)
        base_y += n_track*segment_height + (n_track-1)*row_gap
        base_y += track_gap  # Update base_y for the next joint
        max_y = base_y  # Update max_y to the last base_y

    # Draw headers and segments
    first_joint = True
    active_segments = set()
    for joint, y_data in joint_y_positions.items():
        for track in y_data['rows']:
            row_index = track['track_number']
            y_position = y_data['base_y'] + row_index * (segment_height + row_gap)
            for segment in track['m_list']:
                start_x = timeline_start_x + (segment['start'] / total_duration) * (width - timeline_start_x - 20)
                end_x = timeline_start_x + (segment['end'] / total_duration) * (width - timeline_start_x - 20)
                color = colors[motioncodes.index(segment) % len(colors)]  # Cycle through colors

                # Check if segment should be highlighted
                if segment['start'] <= frame_number <= segment['end']:
                    # Calculate the blinking effect
                    direction = 1 if (frame_number // 10) % 2 == 0 else -1
                    blink_phase = (frame_number % 10) / 10  # Normalize to range [0, 1]
                    if direction == -1:
                        blink_phase = 1-blink_phase
                    blink_phase *= 0.8
                    original_color = np.array(ImageColor.getrgb(color))
                    highlight_color = np.clip(
                        original_color + (np.array([255, 255, 255]) - original_color) * blink_phase, 0, 255)
                    highlight_color = tuple(highlight_color.astype(int))
                else:
                    highlight_color = color

                # Draw the segment
                draw.rectangle([start_x, y_position - segment_height // 2, end_x, y_position + segment_height // 2],
                               fill=highlight_color, outline="black")

                # Track active segments for the legend
                if segment['start'] <= frame_number <= segment['end']:
                    active_segments.add(color)

        # Calculate the middle y position for the header text
        total_rows = len(y_data['rows'])
        # Calculate the total height occupied by the joint's tracks and gaps
        total_height = (total_rows-1 ) * (segment_height + row_gap) + segment_height
        # Calculate the middle y position for the header text
        header_y_position = y_data['base_y'] + total_height / 2 - segment_height / 2 - row_gap - 10
        draw.text((10, header_y_position), joint, fill="black", font=font)
        # draw.text((10,  y_data['base_y']), '1', fill="black", font=font)
        # draw.text((10, header_y_position-row_gap), '2', fill="black", font=font)
        # Skip drawing the separation line for the first joint header
        if not first_joint:
            # Draw a separation line between joint headers
            draw.line([(10, y_data['base_y'] - segment_height // 2 - track_gap // 2),
                       (width - 20, y_data['base_y'] - segment_height // 2 - track_gap // 2)], fill="black",
                      width=4)
        first_joint = False

    # Calculate the base y position for the time ruler
    ruler_base_y = max_y # + track_gap

    # Draw current time indicator (limited height)
    current_x = timeline_start_x + (frame_number / total_duration) * (width - timeline_start_x - 20)
    draw.line([(current_x, 20), (current_x, ruler_base_y+segment_height/2)], fill="red", width=8)

    # Draw the legend
    legend_x = 10  # Start the legend on the left side
    legend_y = max_y + 200  # Place the legend below the last joint track with enough space for the time ruler
    legend_gap = 20
    legend_box_size = segment_height  # Adjusted box size to match segment height

    legend = {}
    for segment in motioncodes:
        color = colors[motioncodes.index(segment) % len(colors)]  # Cycle through colors
        description = f"{segment['spatial']} {segment['temporal']}".strip()
        legend[color] = description

    for color, description in legend.items():
        if color in active_segments:
            # Calculate the blinking effect
            direction = 1 if (frame_number // 10) % 2 == 0 else -1
            blink_phase = (frame_number % 10) / 10  # Normalize to range [0, 1]
            if direction == -1:
                blink_phase = 1 - blink_phase
            blink_phase *= 0.8
            original_color = np.array(ImageColor.getrgb(color))
            highlight_color = np.clip(
                original_color + (np.array([255, 255, 255]) - original_color) * blink_phase, 0, 255)
            highlight_color = tuple(highlight_color.astype(int))
        else:
            highlight_color = color

        # Draw the color box and description in the legend
        draw.rectangle([legend_x, legend_y, legend_x + legend_box_size, legend_y + legend_box_size],
                       fill=highlight_color,
                       outline="black")
        draw.text((legend_x + legend_box_size + 10, legend_y), description, fill="black", font=font)
        legend_y += legend_box_size + legend_gap

    # Draw time ruler as a joint header
    ruler_header_y = ruler_base_y - segment_height/2 + row_gap
    # Calculate the middle y position for the ruler header text
    draw.text((10, ruler_header_y), "Frame", fill="black", font=font)


    # Draw a separation line between the ruler header and other headers
    draw.line([(10, ruler_base_y - segment_height // 2 - track_gap // 2),
               (width - 20, ruler_base_y - segment_height // 2 - track_gap // 2)], fill="black", width=4)
    txt_height = 5
    ruler_line_pos =  ruler_base_y+segment_height/2
    draw.line([(timeline_start_x, ruler_line_pos), (width - 20, ruler_line_pos)], fill="black", width=4)

    frame_interval = total_duration / 100  # Define the frame interval for the ruler

    for frame in range(int(total_duration) + 1):
        frame_x = timeline_start_x + (frame / total_duration) * (width - timeline_start_x - 20)
        if frame % 10 == 0:
            draw.line([(frame_x, ruler_line_pos - 60), (frame_x, ruler_line_pos)], fill="black", width=4)
            font_ruler_small = ImageFont.truetype("arial.ttf", 30)
            draw.text((frame_x - 10, ruler_line_pos + 5), str(frame), fill="black", font=font_ruler_small)
        else:
            draw.line([(frame_x, ruler_line_pos - 40), (frame_x, ruler_line_pos)], fill="black", width=2)

    # Draw a vertical line between headers and multi-track timeline
    draw.line([(timeline_start_x - 10, 10), (timeline_start_x - 10, ruler_base_y + 100)], fill="black", width=4)
    # Draw a box around the whole multi-track timeline
    draw.rectangle([(10, 10), (width - 10, ruler_base_y + 100)], outline="black", width=4)

    return img

def create_timeline_image_with_blinking_good2(motioncodes, frame_number, total_frames):
    width = 2000  # Width for higher resolution
    height = 1500  # Height for more tracks and higher resolution
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 50)  # Font size adjusted
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial.ttf is not available

    # Define colors for the segments
    colors = ["blue", "green", "orange", "purple", "pink", "cyan", "magenta"]

    # Calculate total duration for normalization
    total_duration = total_frames # max(segment['end'] for segment in motioncodes)

    # Organize motion codes by joint
    joint_segments = {}
    for segment in motioncodes:


        # Adjust the joint names
        segment['joint1'] = ' '.join(word.strip().capitalize() for word in segment['joint1'].strip("[]").
                                     replace("'", "").split(',')).replace('None', '')
        segment['joint2'] = ' '.join(word.strip().capitalize() for word in segment['joint2'].strip("[]").
                                     replace("'", "").split(',')).replace('None', '')
        joint = segment['joint1']
        if joint not in joint_segments:
            joint_segments[joint] = []
        joint_segments[joint].append(segment)

    # Calculate maximum width of joint names
    max_joint_name_width = max(
        draw.textbbox((0, 0), joint, font=font)[2] for joint in joint_segments.keys()) + 20
    timeline_start_x = max_joint_name_width + 20  # Start timeline after the header text column

    # Create a dictionary to store y positions and rows for each joint
    joint_y_positions = {}
    org_base_y = 200
    base_y = org_base_y
    segment_height = 100  # Height of each segment adjusted for readability
    row_gap = 20  # Space between rows within a joint track
    track_gap = 20  # Space between tracks of different joints

    # Define multi-track info
    joint2motioncodes = {}
    for mc in motioncodes:
        if mc['joint1'] not in joint2motioncodes:
            joint2motioncodes[mc['joint1']] = []
        joint2motioncodes[mc['joint1']].append(mc)

    multi_track_info = {}
    for joint, j_motioncodes in joint2motioncodes.items():
        multi_track_info[joint] = []
        for mc in j_motioncodes:
            placed = False
            for track in multi_track_info[joint]:
                if all(mc['start'] >= existing_mc['end'] or mc['end'] <= existing_mc['start'] for existing_mc in
                       track['m_list']):
                    track['m_list'].append(mc)
                    placed = True
                    break
            if not placed:
                multi_track_info[joint].append({'track_number': len(multi_track_info[joint]), 'm_list': [mc]})

    # Determine y positions for each joint based on multi-track info
    max_y = base_y
    for joint, tracks in multi_track_info.items():
        joint_y_positions[joint] = {'base_y': base_y, 'rows': tracks}
        n_track = len(tracks)
        base_y += n_track*segment_height + (n_track-1)*row_gap
        base_y += track_gap  # Update base_y for the next joint
        max_y = base_y  # Update max_y to the last base_y


    # ------------
    # Calculate required height for the legend
    legend_box_size = segment_height // 3 # Adjusted box size to match segment height
    legend_gap = 20
    num_legends = len(motioncodes) #len(set(
        # color for segment in motioncodes if segment['start'] <= frame_number <= segment['end'] for color in
        # [colors[motioncodes.index(segment) % len(colors)]]))
    legend_height = num_legends * (legend_box_size + legend_gap)   # Include some space for padding

    # Add space for time ruler
    time_ruler_height = segment_height + 50  # Space for ruler header and line

    # Update the figure height
    height = max_y + legend_height + time_ruler_height + 100

    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # ------------

    font_title = ImageFont.truetype("arial.ttf", 70)
    draw.text((10, 10), "Dynamic Segment Detection Algorithm for Motioncodes", fill="black", font=font_title)

    # Draw headers and segments
    first_joint = True
    s_id = 0
    active_segments = set()
    for joint, y_data in joint_y_positions.items():
        for track in y_data['rows']:
            row_index = track['track_number']
            y_position = y_data['base_y'] + row_index * (segment_height + row_gap)
            for segment in track['m_list']:
                s_id +=1
                start_x = timeline_start_x + (segment['start'] / total_duration) * (width - timeline_start_x - 20)
                end_x = timeline_start_x + (segment['end'] / total_duration) * (width - timeline_start_x - 20)
                color = colors[motioncodes.index(segment) % len(colors)]  # Cycle through colors

                # Check if segment should be highlighted
                if segment['start'] <= frame_number <= segment['end']:
                    # Calculate the blinking effect
                    direction = 1 if (frame_number // 10) % 2 == 0 else -1
                    blink_phase = (frame_number % 10) / 10  # Normalize to range [0, 1]
                    if direction == -1:
                        blink_phase = 1-blink_phase
                    blink_phase *= 0.8
                    original_color = np.array(ImageColor.getrgb(color))
                    highlight_color = np.clip(
                        original_color + (np.array([255, 255, 255]) - original_color) * blink_phase, 0, 255)
                    highlight_color = tuple(highlight_color.astype(int))
                else:
                    highlight_color = color

                # Draw the segment
                draw.rectangle([start_x, y_position - segment_height // 2, end_x, y_position + segment_height // 2],
                               fill=highlight_color, outline="black")
                draw.text(((start_x + end_x)/2 - len(str(s_id))*15, y_position- segment_height // 4), str(s_id), fill='black',
                          font=font)

                # Track active segments for the legend
                if segment['start'] <= frame_number <= segment['end']:
                    active_segments.add(color)

        # Calculate the middle y position for the header text
        total_rows = len(y_data['rows'])
        # Calculate the total height occupied by the joint's tracks and gaps
        total_height = (total_rows-1 ) * (segment_height + row_gap) + segment_height
        # Calculate the middle y position for the header text
        header_y_position = y_data['base_y'] + total_height / 2 - segment_height / 2 - row_gap - 10
        draw.text((15, header_y_position), joint, fill="black", font=font)
        # draw.text((10,  y_data['base_y']), '1', fill="black", font=font)
        # draw.text((10, header_y_position-row_gap), '2', fill="black", font=font)
        # Skip drawing the separation line for the first joint header
        if not first_joint:
            # Draw a separation line between joint headers
            draw.line([(10, y_data['base_y'] - segment_height // 2 - track_gap // 2),
                       (width - 20, y_data['base_y'] - segment_height // 2 - track_gap // 2)], fill="black",
                      width=4)
        first_joint = False

    # Calculate the base y position for the time ruler
    ruler_base_y = max_y # + track_gap

    # Draw current time indicator (limited height)
    current_x = timeline_start_x + (frame_number / total_duration) * (width - timeline_start_x - 20)
    draw.line([(current_x, 20), (current_x, ruler_base_y+segment_height/2)], fill="red", width=8)

    # Draw the legend
    legend_x = 10  # Start the legend on the left side
    legend_y = max_y + 150  # Place the legend below the last joint track with enough space for the time ruler
    # legend_gap = 20
    # legend_box_size = segment_height // 2  # Adjusted box size to match segment height

    legend = {}
    for segment in motioncodes:
        color = colors[motioncodes.index(segment) % len(colors)]  # Cycle through colors
        action = ' '.join(segment['spatial'].split('_'))
        velocity = ' '.join(segment['temporal'].split('_'))
        if velocity=='': velocity='Ignored'
        joint_str = segment['joint1'] + "   " + segment['joint2']
        # description = f"Joint: {joint_str:<25} Spatial: {action:<30} Tempral: {velocity:<10}"
        description = [f"Joint: {joint_str}", f"Spatial: {action}", f"Velocity: {velocity}"]
        legend[len(legend)] = (color, description)
        print(description)

    # for color, description in legend.items():
    legend_texts = []
    for id, val in legend.items():
        color, description = val
        if color in active_segments:
            # Calculate the blinking effect
            direction = 1 if (frame_number // 10) % 2 == 0 else -1
            blink_phase = (frame_number % 10) / 10  # Normalize to range [0, 1]
            if direction == -1:
                blink_phase = 1 - blink_phase
            blink_phase *= 0.8
            original_color = np.array(ImageColor.getrgb(color))
            highlight_color = np.clip(
                original_color + (np.array([255, 255, 255]) - original_color) * blink_phase, 0, 255)
            highlight_color = tuple(highlight_color.astype(int))
        else:
            highlight_color = color

        # Draw the color box and description in the legend
        draw.rectangle([legend_x, legend_y, legend_x + 2*legend_box_size, legend_y + legend_box_size],
                       fill=highlight_color,
                       outline="black")
        font_legend_small = ImageFont.truetype("arial.ttf", 30)
        draw.text((legend_x + legend_box_size/2 , legend_y), str(id+1), fill='black',
                  font=font_legend_small)

        # description = description.replace(' ', '*')


        draw.text((legend_x + legend_box_size * 2 + 10, legend_y), description[0], fill=highlight_color,
                  font=font_legend_small)
        draw.text((legend_x + legend_box_size * 2 + 700, legend_y), description[1], fill=highlight_color,
                  font=font_legend_small)
        draw.text((legend_x + legend_box_size * 2 + 1400, legend_y), description[2], fill=highlight_color,
                  font=font_legend_small)

        # draw.text((legend_x + legend_box_size*2 + 10, legend_y), f'{description:<100}', fill="black", font=font_legend_small)

        legend_y += legend_box_size + legend_gap


    # Draw time ruler as a joint header
    ruler_header_y = ruler_base_y - segment_height/2 + row_gap
    # Calculate the middle y position for the ruler header text
    draw.text((10, ruler_header_y), "Frame", fill="black", font=font)


    # Draw a separation line between the ruler header and other headers
    draw.line([(10, ruler_base_y - segment_height // 2 - track_gap // 2),
               (width - 20, ruler_base_y - segment_height // 2 - track_gap // 2)], fill="black", width=4)
    txt_height = 5
    ruler_line_pos =  ruler_base_y+segment_height/2
    draw.line([(timeline_start_x, ruler_line_pos), (width - 20, ruler_line_pos)], fill="black", width=4)

    frame_interval = total_duration / 100  # Define the frame interval for the ruler

    for frame in range(int(total_duration) + 1):
        frame_x = timeline_start_x + (frame / total_duration) * (width - timeline_start_x - 20)
        if frame % 10 == 0:
            draw.line([(frame_x, ruler_line_pos - 60), (frame_x, ruler_line_pos)], fill="black", width=4)
            font_ruler_small = ImageFont.truetype("arial.ttf", 30)
            draw.text((frame_x - 10, ruler_line_pos + 5), str(frame), fill="black", font=font_ruler_small)
        else:
            draw.line([(frame_x, ruler_line_pos - 40), (frame_x, ruler_line_pos)], fill="black", width=2)

    # Draw a vertical line between headers and multi-track timeline
    draw.line([(timeline_start_x - 10, org_base_y-segment_height//2 - row_gap), (timeline_start_x - 10, ruler_base_y + 100)], fill="black", width=4)
    # Draw a box around the whole multi-track timeline
    draw.rectangle([(10, org_base_y-segment_height//2 - row_gap), (width - 10, ruler_base_y + 100)], outline="black", width=4)

    return img

from collections import OrderedDict
def create_timeline_image_with_blinking(motioncodes, frame_number, total_frames):

    width = 2000  # Width for higher resolution
    height = 1500  # Height for more tracks and higher resolution
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 50)  # Font size adjusted
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial.ttf is not available

    # Define colors for the segments
    colors = ["blue", "green", "orange", "purple", "pink", "cyan", "magenta"]

    # Calculate total duration for normalization
    total_duration = total_frames # max(segment['end'] for segment in motioncodes)

    # Organize motion codes by joint
    joint_segments = {}
    for segment in motioncodes:


        # Adjust the joint names
        segment['joint1'] = ' '.join(word.strip().capitalize() for word in segment['joint1'].strip("[]").
                                     replace("'", "").split(',')).replace('None', '')
        segment['joint2'] = ' '.join(word.strip().capitalize() for word in segment['joint2'].strip("[]").
                                     replace("'", "").split(',')).replace('None', '')
        joint = segment['joint1']
        if joint not in joint_segments:
            joint_segments[joint] = []
        joint_segments[joint].append(segment)

    # Calculate maximum width of joint names
    max_joint_name_width = max(
        draw.textbbox((0, 0), joint, font=font)[2] for joint in joint_segments.keys()) + 20
    timeline_start_x = max_joint_name_width + 20  # Start timeline after the header text column

    # Create a dictionary to store y positions and rows for each joint
    joint_y_positions = {}
    org_base_y = 200
    base_y = org_base_y
    segment_height = 100  # Height of each segment adjusted for readability
    row_gap = 20  # Space between rows within a joint track
    track_gap = 20  # Space between tracks of different joints

    # Define multi-track info
    joint2motioncodes = {}
    for mc in motioncodes:
        if mc['joint1'] not in joint2motioncodes:
            joint2motioncodes[mc['joint1']] = []
        joint2motioncodes[mc['joint1']].append(mc)

    multi_track_info = {}
    for joint, j_motioncodes in joint2motioncodes.items():
        multi_track_info[joint] = []
        for mc in j_motioncodes:
            placed = False
            for track in multi_track_info[joint]:
                if all(mc['start'] >= existing_mc['end'] or mc['end'] <= existing_mc['start'] for existing_mc in
                       track['m_list']):
                    track['m_list'].append(mc)
                    placed = True
                    break
            if not placed:
                multi_track_info[joint].append({'track_number': len(multi_track_info[joint]), 'm_list': [mc]})

    # Determine y positions for each joint based on multi-track info
    max_y = base_y
    for joint, tracks in multi_track_info.items():
        joint_y_positions[joint] = {'base_y': base_y, 'rows': tracks}
        n_track = len(tracks)
        base_y += n_track*segment_height + (n_track-1)*row_gap
        base_y += track_gap  # Update base_y for the next joint
        max_y = base_y  # Update max_y to the last base_y


    # ------------
    # Calculate required height for the legend
    legend_box_size = segment_height // 3 # Adjusted box size to match segment height
    legend_gap = 20
    num_legends = len(motioncodes) #len(set(
        # color for segment in motioncodes if segment['start'] <= frame_number <= segment['end'] for color in
        # [colors[motioncodes.index(segment) % len(colors)]]))
    legend_height = num_legends * (legend_box_size + legend_gap)   # Include some space for padding

    # Add space for time ruler
    time_ruler_height = segment_height + 50  # Space for ruler header and line

    # Update the figure height
    height = max_y + legend_height + time_ruler_height + 100

    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # ------------

    font_title = ImageFont.truetype("arial.ttf", 70)
    draw.text((10, 10), "Dynamic Segment Detection Algorithm for Motincodes", fill="black", font=font_title)

    # Draw headers and segments
    first_joint = True
    s_id = 0
    active_segments = set()
    segment_colors = {}
    segment_ids2index = {}
    for joint, y_data in joint_y_positions.items():
        for track in y_data['rows']:
            row_index = track['track_number']
            y_position = y_data['base_y'] + row_index * (segment_height + row_gap)
            for segment in track['m_list']:
                s_id +=1
                start_x = timeline_start_x + (segment['start'] / total_duration) * (width - timeline_start_x - 20)
                end_x = timeline_start_x + (segment['end'] / total_duration) * (width - timeline_start_x - 20)


                # color = colors[motioncodes.index(segment) % len(colors)]  # Cycle through colors
                segment_id = (segment['spatial'], segment['start'], segment['end'], segment['joint1'], segment['joint2'])
                segment_ids2index[segment_id] = s_id
                if segment_id not in segment_colors:
                    segment_colors[segment_id] = colors[len(segment_colors) % len(colors)]
                color = segment_colors[segment_id]




                # Check if segment should be highlighted
                if segment['start'] <= frame_number <= segment['end']:
                    # Calculate the blinking effect
                    direction = 1 if (frame_number // 10) % 2 == 0 else -1
                    blink_phase = (frame_number % 10) / 10  # Normalize to range [0, 1]
                    if direction == -1:
                        blink_phase = 1-blink_phase
                    blink_phase *= 0.8
                    original_color = np.array(ImageColor.getrgb(color))
                    highlight_color = np.clip(
                        original_color + (np.array([255, 255, 255]) - original_color) * blink_phase, 0, 255)
                    highlight_color = tuple(highlight_color.astype(int))
                else:
                    highlight_color = color

                # Draw the segment
                draw.rectangle([start_x, y_position - segment_height // 2, end_x, y_position + segment_height // 2],
                               fill=highlight_color, outline="black")
                draw.text(((start_x + end_x)/2 - len(str(s_id))*15, y_position- segment_height // 4), str(s_id), fill='black',
                          font=font)

                # Track active segments for the legend
                if segment['start'] <= frame_number <= segment['end']:
                    active_segments.add(color)

        # Calculate the middle y position for the header text
        total_rows = len(y_data['rows'])
        # Calculate the total height occupied by the joint's tracks and gaps
        total_height = (total_rows-1 ) * (segment_height + row_gap) + segment_height
        # Calculate the middle y position for the header text
        header_y_position = y_data['base_y'] + total_height / 2 - segment_height / 2 - row_gap - 10
        draw.text((15, header_y_position), joint, fill="black", font=font)
        # draw.text((10,  y_data['base_y']), '1', fill="black", font=font)
        # draw.text((10, header_y_position-row_gap), '2', fill="black", font=font)
        # Skip drawing the separation line for the first joint header
        if not first_joint:
            # Draw a separation line between joint headers
            draw.line([(10, y_data['base_y'] - segment_height // 2 - track_gap // 2),
                       (width - 20, y_data['base_y'] - segment_height // 2 - track_gap // 2)], fill="black",
                      width=4)
        first_joint = False

    # Calculate the base y position for the time ruler
    ruler_base_y = max_y # + track_gap

    # Draw current time indicator (limited height)
    current_x = timeline_start_x + (frame_number / total_duration) * (width - timeline_start_x - 20)
    draw.line([(current_x, org_base_y-segment_height//2 - row_gap+ 5), (current_x, ruler_base_y+segment_height/2)], fill="red", width=10)

    # Draw the legend
    legend_x = 10  # Start the legend on the left side
    legend_y = max_y + 150  # Place the legend below the last joint track with enough space for the time ruler
    # legend_gap = 20
    # legend_box_size = segment_height // 2  # Adjusted box size to match segment height

    legend = {}
    index2sort_legend = 0
    for segment in motioncodes:

        # color = colors[motioncodes.index(segment) % len(colors)]  # Cycle through colors
        segment_id = (segment['spatial'], segment['start'], segment['end'], segment['joint1'], segment['joint2'])
        color = segment_colors[segment_id]

        action = ' '.join(segment['spatial'].split('_'))
        velocity = ' '.join(segment['temporal'].split('_'))
        if velocity=='': velocity='Ignored'
        joint_str = segment['joint1'] + "   " + segment['joint2']
        # description = f"Joint: {joint_str:<25} Spatial: {action:<30} Tempral: {velocity:<10}"
        description = [f" {joint_str}", f"Spatial: {action}", f"Velocity: {velocity}"]
        legend[segment_id] = ( color, description)
        print(description)

    # for color, description in legend.items():
    legend_texts = []
    rev_segment_ids2index = {v: k for k, v in segment_ids2index.items()}
    for current_s_index_sorted in range(len(segment_ids2index.items())):
    # for segment_id, val in legend.items():
        segment_id = rev_segment_ids2index[current_s_index_sorted+1]
        color, description = legend[segment_id]
        # color, description = val
        if color in active_segments:
            # Calculate the blinking effect
            direction = 1 if (frame_number // 10) % 2 == 0 else -1
            blink_phase = (frame_number % 10) / 10  # Normalize to range [0, 1]

            if direction == -1:
                blink_phase = 1 - blink_phase

            font_blink_size = 0.5 + 0.5 * blink_phase
            blink_phase *= 0.8
            original_color = np.array(ImageColor.getrgb(color))
            highlight_color = np.clip(
                original_color + (np.array([255, 255, 255]) - original_color) * blink_phase, 0, 255)
            highlight_color = tuple(highlight_color.astype(int))
            font_highlight_color =  tuple(np.clip((np.array([255, 255, 255]) ) * blink_phase, 0, 255).astype(int))
            font_legend_small = ImageFont.truetype("arial.ttf" if direction==1 else "arial.ttf", 30)# decided to make no difference.
        else:
            highlight_color = color
            font_highlight_color = tuple(np.array([0, 0, 0]).astype(int))
            font_legend_small = ImageFont.truetype("arial.ttf", 30)

        # Draw the color box and description in the legend
        draw.rectangle([legend_x, legend_y, legend_x + 2*legend_box_size, legend_y + legend_box_size],
                       fill=highlight_color,
                       outline="black")

        # draw.text((legend_x + legend_box_size/2 , legend_y), str(id+1), fill='black',
        #           font=font_legend_small)
        draw.text((legend_x + legend_box_size / 2, legend_y), str(current_s_index_sorted+1), fill='black',
                  font=font_legend_small)

        # description = description.replace(' ', '*')


        draw.text((legend_x + legend_box_size * 2 + 10, legend_y), description[0], fill=font_highlight_color,
                  font=font_legend_small)
                  # stroke_width= 0 if color in active_segments and direction==1 else 0, stroke_fill="black")
        draw.text((legend_x + legend_box_size * 2 + 700, legend_y), description[1], fill=font_highlight_color,
                  font=font_legend_small)
                  # stroke_width= 0 if color in active_segments and direction==1 else 0, stroke_fill="black")
        draw.text((legend_x + legend_box_size * 2 + 1400, legend_y), description[2], fill=font_highlight_color,
                  font=font_legend_small)
                  # stroke_width= 0 if color in active_segments and direction==1 else 0, stroke_fill="black")

        # draw.text((legend_x + legend_box_size*2 + 10, legend_y), f'{description:<100}', fill="black", font=font_legend_small)

        legend_y += legend_box_size + legend_gap


    # Draw time ruler as a joint header
    ruler_header_y = ruler_base_y - segment_height/2 + row_gap
    # Calculate the middle y position for the ruler header text
    draw.text((10, ruler_header_y), "Frame", fill="black", font=font)


    # Draw a separation line between the ruler header and other headers
    draw.line([(10, ruler_base_y - segment_height // 2 - track_gap // 2),
               (width - 20, ruler_base_y - segment_height // 2 - track_gap // 2)], fill="black", width=4)
    txt_height = 5
    ruler_line_pos =  ruler_base_y+segment_height/2
    draw.line([(timeline_start_x, ruler_line_pos), (width - 20, ruler_line_pos)], fill="black", width=4)

    frame_interval = total_duration / 100  # Define the frame interval for the ruler

    for frame in range(int(total_duration) + 1):
        frame_x = timeline_start_x + (frame / total_duration) * (width - timeline_start_x - 20)
        if frame % 10 == 0:
            draw.line([(frame_x, ruler_line_pos - 60), (frame_x, ruler_line_pos)], fill="black", width=4)
            font_ruler_small = ImageFont.truetype("arial.ttf", 30)
            draw.text((frame_x - 10, ruler_line_pos + 5), str(frame), fill="black", font=font_ruler_small)
        else:
            draw.line([(frame_x, ruler_line_pos - 40), (frame_x, ruler_line_pos)], fill="black", width=2)

    # Draw a vertical line between headers and multi-track timeline
    draw.line([(timeline_start_x - 10, org_base_y-segment_height//2 - row_gap), (timeline_start_x - 10, ruler_base_y + 100)], fill="black", width=4)
    # Draw a box around the whole multi-track timeline
    draw.rectangle([(10, org_base_y-segment_height//2 - row_gap), (width - 10, ruler_base_y + 100)], outline="black", width=4)

    return img



def create_gif_with_blinking(motioncodes, total_frames, outname):
    images = []
    for frame_number in range(total_frames):
        #current_time = (frame_number / total_frames) * max(segment['end'] for segment in motioncodes)
        img = create_timeline_image_with_blinking(motioncodes, frame_number, total_frames)
        images.append(img)

    # Save as GIF with a smooth transition
    images[0].save(outname, save_all=True, append_images=images[1:], optimize=False, duration=100,
                   loop=0)


# Test case
# if __name__ == "__main__":
    # Example motioncodes for testing
    # motioncodes = [
    #     {'joint1': 'Hip', 'start': 0, 'end': 10, 'spatial': 'Move', 'temporal': 'Forward'},
    #     {'joint1': 'Knee', 'start': 5, 'end': 15, 'spatial': 'Bend', 'temporal': 'Backward'},
    #     {'joint1': 'Elbow', 'start': 10, 'end': 20, 'spatial': 'Raise', 'temporal': 'Upward'}
    # ]

    # create_gif_with_blinking(motioncodes, total_frames=100)

from PIL import Image, ImageSequence
def merge_gifs_side_by_side(gif1_path, gif2_path, output_path, default_duration=100):
    # Open the GIFs
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    # Check if both GIFs have the same number of frames
    frames_gif1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    frames_gif2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]

    if len(frames_gif1) != len(frames_gif2):
        raise ValueError("GIFs do not have the same number of frames")

    # Get dimensions
    width1, height1 = gif1.size
    width2, height2 = gif2.size

    # Create new frames by merging corresponding frames side by side
    new_frames = []
    for frame1, frame2 in zip(frames_gif1, frames_gif2):
        new_frame = Image.new('RGBA', (width1 + width2, max(height1, height2)))
        new_frame.paste(frame1, (0, 0))
        new_frame.paste(frame2, (width1, 0))
        new_frames.append(new_frame)

    duration = gif1.info.get('duration', default_duration) or default_duration

    # Save the new frames as a new GIF
    new_frames[0].save(
        output_path,
        save_all=True,
        append_images=new_frames[1:],
        loop=0,
        duration=duration
    )