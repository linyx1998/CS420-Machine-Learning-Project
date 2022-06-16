import matplotlib.pyplot as plt
# valid set result
82.182404
84.118401
84.809601
84.900803
84.588799
84.084801
83.801598
83.593597
83.462402
83.443199
y_accu_vgg19 = [(81.7200), (83.7120), (84.6768), (84.8800), (84.5104), (84.2752), (83.7312), (83.2464), (83.1072), (82.9840)]

def mydraw(x, y, xl='epoch', yl = 'Accuracy'):
    plt.plot(x, y, marker='o', markersize=3)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend(['test set', 'train set'])
    plt.show()

if(__name__=='__main__'):
    x = range(1,11)
    y_accu_test = [82.289597, 84.318398, 84.875198, 84.897598, 84.876801, 84.094398, 84.075203, 83.662399, 83.684799, 83.537598]
    y_accu_valid = [82.182404,84.118401,84.809601,84.900803,84.588799,84.084801,83.801598,83.593597,83.462402,83.443199]

    y_accu_vgg19 = [(81.7200), (83.7120), (84.6768), (84.8800), (84.5104), (84.2752), (83.7312), (83.2464), (83.1072), (82.9840)]
    plt.plot(x, y_accu_vgg19, marker='o', markersize=3)
    plt.plot(x, y_accu_valid, marker='o', markersize=3)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['vgg19, valid set', 'vgg16, valid set'])
    plt.title("Accuracy, different epoch, VGG16 and VGG19")
    plt.show()