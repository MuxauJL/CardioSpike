

# From https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703
def make_weights_for_balanced_classes(labels, nclasses):
    count = [0] * nclasses
    for item in labels:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight
