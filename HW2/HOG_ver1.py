import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():

    '''
    Input : None
    Output :
        filter_x : 2D filter (3 x 3)
        filter_y : 2D filter (3 x 3)
    '''

    filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    filter_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    return filter_x, filter_y


def filter_image(im, filter):

    '''
    Input : 
        im : 2D image (m x n)
        filter : 2D filter (k x k)
    Output :
        im_filtered : 2D image (m x n)
    '''
    
    k = filter.shape[0]
    pad_size = k // 2

    im_padded = np.pad(im, (pad_size, pad_size), 'constant')
    im_filtered = np.zeros(im.shape)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im_filtered[i, j] = np.sum(im_padded[i:i+k, j:j+k] * filter)

    return im_filtered


def get_gradient(im_dx, im_dy):

    '''
    Input : 
        im_dx : 2D image (m x n)
        im_dy : 2D image (m x n)
    Output :
        grad_mag : 2D image (m x n)
        grad_angle : 2D image (m x n)
    '''

    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    grad_angle = np.arctan2(im_dy, im_dx)

    grad_angle[grad_angle < 0] += np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):

    '''
    Input : 
        grad_mag : 2D image (m x n)
        grad_angle : 2D image (m x n)
        cell_size : int
    Output :
        ori_histo : 3D array (M x N x 6)
    '''

    m = grad_mag.shape[0]
    n = grad_mag.shape[1]

    M = m // cell_size
    N = n // cell_size
    
    ori_histo = np.zeros((M, N, 6))

    for i in range(M):
        for j in range(N):
            row_idx_start = i * cell_size
            col_idx_start = j * cell_size

            for k in range(cell_size):
                for l in range(cell_size):
                    row_idx = row_idx_start + k
                    col_idx = col_idx_start + l
                    
                    angle = grad_angle[row_idx, col_idx]
                    mag = grad_mag[row_idx, col_idx]

                    if    0 <= angle < np.pi/12 or 11*np.pi/12 <= angle < np.pi:    ori_histo[i, j, 0] += mag
                    elif np.pi/12   <= angle < np.pi/4:                             ori_histo[i, j, 1] += mag
                    elif np.pi/4    <= angle < 5*np.pi/12:                          ori_histo[i, j, 2] += mag
                    elif 5*np.pi/12 <= angle < 7*np.pi/12:                          ori_histo[i, j, 3] += mag
                    elif 7*np.pi/12 <= angle < 3*np.pi/4:                           ori_histo[i, j, 4] += mag
                    elif 3*np.pi/4  <= angle < 11*np.pi/12:                         ori_histo[i, j, 5] += mag

    return ori_histo


def get_block_descriptor(ori_histo, block_size):

    '''
    Input : 
        ori_histo : 3D array (M x N x 6)
        block_size : int
    Output :
        ori_histo_normalized : 3D array ((M - block_size + 1) x (N - block_size + 1) x (6 x block_size**2))
    '''
    
    M = ori_histo.shape[0]
    N = ori_histo.shape[1]

    ori_histo_normalized = np.zeros((M - block_size + 1, N - block_size + 1, 6 * block_size**2))

    for i in range(M - block_size + 1):
        for j in range(N - block_size + 1):
            block_ori_histo = ori_histo[i:i+block_size, j:j+block_size, :].flatten()
            ori_histo_normalized[i, j, :] = block_ori_histo / np.sqrt(np.sum(block_ori_histo**2) + (1e-3)**2)

    return ori_histo_normalized


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='red', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    # plt.show()
    plt.savefig('hog.png')


def extract_hog(im, visualize=False, cell_size=8, block_size=2):

    '''
    Input : 
        im : 2D image (m x n)
        visualize : bool
        cell_size : int
        block_size : int
    Output :
        hog : 1D array
    '''
    # normalize image
    im = im.astype('float') / 255.0

    # get gradient
    filter_x, filter_y = get_differential_filter()

    # filter image
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)

    # get gradient magnitude and angle
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)

    # build histogram
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)

    # get block descriptor
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size)
    hog = ori_histo_normalized.flatten()

    if visualize:
        visualize_hog(im, hog, cell_size, block_size)

    return hog


def IoU(box1, box2, box_size):

    '''
    Input : 
        box1, box2: 1D array (1 x 3) [x, y, score]
        box_size: 1D array (1 x 2) [template_x, template_y]
    Output :
        iou : float
    '''

    x1, y1, ncc1 = box1
    x2, y2, ncc2 = box2
    
    box1_coords = [x1, y1, x1 + box_size[0], y1 + box_size[1]]
    box2_coords = [x2, y2, x2 + box_size[0], y2 + box_size[1]]
    
    # Intersection area
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # no overlap
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Union area
    box1_area = box_size[0] * box_size[1]
    box2_area = box_size[0] * box_size[1]
    union = box1_area + box2_area - intersection
    
    iou = intersection / union
    
    return iou
    

def NCC(hog_target, hog_template):

    '''
    Input :
        hog_target : 1D array
        hog_template : 1D array
    Output :
        ncc : float
    '''

    a = hog_target - np.mean(hog_target)
    b = hog_template - np.mean(hog_template)

    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    return numerator / denominator if denominator != 0 else 0.0


def NMS(bounding_boxes, box_size, iou_threshold = 0.5):
    '''
    Input :
        bounding_boxes : 2D array (k x 3) (x, y, ncc)
        iou_threshold : float
    Output :
        bounding_boxes : 2D array (k x 3)
    '''
    
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[2], reverse=True)

    filtered_boxes = []
    for box in bounding_boxes:
        if len(filtered_boxes) == 0:
            filtered_boxes.append(box)
        else:
            for filtered_box in filtered_boxes:
                if IoU(box, filtered_box, box_size) > iou_threshold:
                    break
            else:
                filtered_boxes.append(box)
    filtered_boxes = np.array(filtered_boxes)
    
    return filtered_boxes


def face_recognition(I_target, I_template):

    '''
    Input : 
        I_target : 2D image (M x N)
        I_template : 2D image (m x n)
    Output :
        bounding_boxes : 2D array (k x 3)
    '''

    target_y = I_target.shape[0]
    target_x = I_target.shape[1]
    template_y = I_template.shape[0]
    template_x = I_template.shape[1]
    
    template_hog = extract_hog(I_template)

    # NCC
    bounding_boxes = []
    for i in range(target_y - template_y + 1):
        for j in range(target_x - template_x + 1):
            target_hog = extract_hog(I_target[i:i+template_y, j:j+template_x])
            ncc = NCC(target_hog, template_hog)
            #threshold 0.48
            if ncc > 0.48:
                bounding_boxes.append([j, i, ncc])

    # NMS
    box_size = [template_x, template_y]
    bounding_boxes = NMS(bounding_boxes, box_size)

    return bounding_boxes


def visualize_face_detection(I_target, bounding_boxes, box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.imsave('result_face_detection.png', fimg, vmin=0, vmax=1)
    plt.show()

if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)

    hog = extract_hog(im, visualize=True)

    I_target= cv2.imread('target.png', 0) # MxN image
    
    I_template = cv2.imread('template.png', 0) # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png') # MxN image (just for visualization)
    
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0]) # visualization code


