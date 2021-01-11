import timeit

from convert_tensorflow import create_training_model
import numpy as np
import pickle
import cv2

IMAGE_SIZE = (112, 112)
BATCH_SUZE = 32


def main():
    model = create_training_model(IMAGE_SIZE, [3, 4, 6, 3], 1, mode='infer')

    model.load_weights('checkpoints/e_500.ckpt')

    def embedding_fn(img1, img2):
        result1 = model.predict(np.expand_dims(img1, axis=0))[0]
        result2 = model.predict(np.expand_dims(img2, axis=0))[0]
        return result1, result2

    test_lfw('dataset/lfw.bin', embedding_fn, IMAGE_SIZE, is_plot=False)


def load_bin(bin_path, input_size):
    import mxnet as mx

    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    first_imgs = np.empty((len(issame_list) * 2, input_size[0], input_size[1], 3))
    second_imgs = np.empty((len(issame_list) * 2, input_size[0], input_size[1], 3))

    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = img - 127.5
        img = img * 0.0078125
        for flip in [0, 1]:
            if flip == 1:
                img = np.fliplr(img)
            if i % 2 == 0:
                first_imgs[i + flip, ...] = img
            else:
                second_imgs[i - 1 + flip, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)

    return first_imgs, second_imgs, np.repeat(issame_list, 2)


def test_lfw(path, embedding_fn, shape, is_plot=False, ver_type='euclidean'):
    ver_dataset = load_bin(path, shape)

    val_acc, val_thr, tp, fp, fn, tn, tpr, fpr = ver_tfrecord(ver_dataset, embedding_fn, ver_type=ver_type)
    print('test accuracy is: %.3f, thr: %.2f, prec: %.3f, rec: %.3f.' %
          (val_acc, val_thr, float(tp) / (tp + fp), float(tp) / (tp + fn)))

    if is_plot:
        plot_roc(fpr, tpr)


def ver_tfrecord(data_set, embedding_fn, verbose=False, ver_type='euclidean'):
    first_list, second_list, true_same = data_set[0], data_set[1], np.array(data_set[2])
    total = len(true_same)
    same = int(np.sum(true_same))
    diff = total - same
    if verbose:
        print('samples: %d, same: %d, diff: %d' % (total, same, diff))

    val_list = []
    start = timeit.default_timer()
    for idx, (first, second) in enumerate(zip(first_list, second_list)):
        result1, result2 = embedding_fn(first, second)

        if ver_type == 'euclidean':
            val = np.linalg.norm(result1 - result2)
            val_list.append(val)
        elif ver_type == 'cosine':
            val = np.dot(result1, np.transpose(result2)) / (np.sqrt(np.dot(result1, np.transpose(result1))) * np.sqrt(
                np.dot(result2, np.transpose(result2))))
            val_list.append((val[0][0] + 1) / 2)
        else:
            raise RuntimeError(f'ver_type: {ver_type} is not exist.')
        if (idx % 1000 == 0) & verbose:
            print('complete %d pairs' % idx)
    if verbose:
        print('cost_times: %.2f sec' % (timeit.default_timer() - start))

    if ver_type == 'euclidean':
        thresholds = np.arange(0.05, 0.5, 0.05)
    elif ver_type == 'cosine':
        thresholds = np.arange(0.05, 1.0, 0.05)
    else:
        raise RuntimeError(f'ver_type: {ver_type} is not exist.')

    accs = []
    tps = []
    fps = []
    fns = []
    tns = []
    for threshold in thresholds:
        if ver_type == 'euclidean':
            pred_same = np.less(val_list, threshold)
        elif ver_type == 'cosine':
            pred_same = np.greater(val_list, threshold)
        else:
            raise RuntimeError(f'ver_type: {ver_type} is not exist.')
        tp = np.sum(np.logical_and(pred_same, true_same))
        tn = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(true_same)))
        fp = diff - tn
        fn = same - tp
        acc = float(tp + tn) / total
        accs.append(acc)
        tps.append(int(tp))
        tns.append(int(tn))
        fps.append(int(fp))
        fns.append(int(fn))
    print('thresholds:', ", ".join("%.2f" % f for f in thresholds))
    print('accs:', ", ".join("%.2f" % f for f in accs))
    tpr = [0 if (tp + fn == 0) else float(tp) / float(tp + fn) for tp, fn in zip(tps, fns)]
    fpr = [0 if (fp + tn == 0) else float(fp) / float(fp + tn) for fp, tn in zip(fps, tns)]
    best_index = int(np.argmax(accs))

    print('tps:', ", ".join("%.2f" % f for f in tps))
    print('tns:', ", ".join("%.2f" % f for f in tns))
    print('fps:', ", ".join("%.2f" % f for f in fps))
    print('fns:', ", ".join("%.2f" % f for f in fns))

    return accs[best_index], thresholds[best_index], tps[best_index], fps[best_index], fns[best_index], tns[
        best_index], tpr, fpr


def plot_roc(fpr, tpr):
    import matplotlib.pyplot as plt
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
