
MEDIAN = 624.0
O_25 = 548.0
O_75 = 732.0
def norm(data):
    data['ampl'] = (data['ampl'] - MEDIAN) / (O_75 - O_25)
    return data

def get_test_transform(opt):
    #return lambda x: x
    return norm


def get_train_transform(opt):
    #return lambda x: x
    return norm