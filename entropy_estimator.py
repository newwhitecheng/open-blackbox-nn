import numpy as np
import kde

def cal_binned_MI(input_x, layer, bin_levels)
    def get_probs(x):
        '''
        x should be (nums of samples, pixels)
        '''
        unique_vector = np.ascontiguousarray(x).view(np.dtype((np.void, np.dtype.itemsize * x.shape[1]))
        _, unique_count, unique_inverse = np.unique(unique_vector, return_index=False, return_count=True, return_inverse=True )
        probs = unique_count / float(np.sum(unique_count))
        return probs, unique_inverse

    #Input is digitized already
    p_x, unique_inverse_x = get_probs(input_x)

    bins = np.linspace(-1, 1, bin_levels, dtype='float32')
    discrited = bins[np.digitize(np.squeeze(layer.reshape(1, -1)), bins)].reshape(layer.shape[0], -1)
    p_t, _ = get_probs(discrited)

    H_T = -np.sum(p_t * np.log(p_t))
    H_T_GIVEN_X = 0.
    for x in unique_inverse_x:
        p_t_given_x, _ = get_probs(digitized[unique_inverse_x == x, :])
        H_LAYER_GIVEN_INPUT += - p_x[x] * np.sum(p_t_given_x * np.log(p_t_given_x))

    return H_LAYER - H_LAYER_GIVEN_INPUT
