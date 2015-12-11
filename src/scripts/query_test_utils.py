
def compute_average_precision(results, correct_ids):
    precisions = [0.0] * len(results)
    cur_ind = 0
    num_correct = 0
    while cur_ind < len(results):
        prev_ind = cur_ind
        cur_score = results[cur_ind][0]
        while cur_ind < len(results) and results[cur_ind][0] >= cur_score:
            if results[cur_ind][1] in correct_ids:
                num_correct += 1
            cur_ind += 1
        
        precision = float(num_correct) / cur_ind
        for i in xrange(prev_ind, cur_ind):
            precisions[i] = precision
    
    average_precision_sum = 0.0
    for i in xrange(len(results)):
        if results[i][1] in correct_ids:
            average_precision_sum += precisions[i]
    
    if len(correct_ids) > 0:
        return average_precision_sum / len(correct_ids)
    else:
        return 0

def compute_11point_precision(results, correct_ids):
    precision = []
    recall = []

    num_correct = 0
    for i in xrange(len(results)):
        if results[i][1] in correct_ids:
            num_correct += 1
        
        precision.append(float(num_correct) / (i + 1))
        if len(correct_ids) > 0:
            recall.append(float(num_correct) / len(correct_ids))
        else:
            recall.append(0)

    interpolated_precision = [x for x in precision]
    max_precision = 0
    for i in xrange(len(precision)):
        ind = len(precision) - (i + 1)
        max_precision = max(max_precision, precision[ind])
        interpolated_precision[ind] = max_precision

    point_precision = []
    for point in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        found_point = False
        for i in xrange(len(recall)):
            if recall[i] >= point:
                point_precision.append(interpolated_precision[i])
                found_point = True
                break
        if not found_point:
            point_precision.append(0.0)

    return point_precision
