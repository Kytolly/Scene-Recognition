import numpy as np
import matplotlib
# from skimage.io import imread
from PIL import Image
from skimage.color import rgb2grey
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
import scipy.stats


def get_tiny_images(image_paths):
    '''
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.

    To build a tiny image feature, resize the original image to a very small
    square resolution (e.g. 16x16). You can either resize the images to square
    while ignoring their aspect ratio, or you can crop the images into squares
    first and then resize evenly. Normalizing these tiny images will increase
    performance modestly.

    As you may recall from class, naively downsizing an image can cause
    aliasing artifacts that may throw off your comparisons. See the docs for
    skimage.transform.resize for details:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

    Suggested functions: skimage.transform.resize, skimage.color.rgb2grey,
                         skimage.io.imread, np.reshape, PIL.Image.open, np.std ...
    '''

    tiny_images_features = []
    img_size = (16, 16)

    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            # Convert to grayscale
            gray_img = rgb2grey(np.array(img))
            # Resize the image
            resized_img = resize(gray_img, img_size, anti_aliasing=True)
            # Flatten the image
            flattened_img = resized_img.reshape(-1)
            # Normalize the feature vector (mean=0, std=1)
            mean = np.mean(flattened_img)
            std = np.std(flattened_img)
            normalized_img = (flattened_img - mean) / (std + 1e-6) # Add epsilon to avoid division by zero

            tiny_images_features.append(normalized_img)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Optionally append a zero vector or handle error differently
            # For now, just skip and print error

    return np.array(tiny_images_features)

def build_vocabulary(image_paths, vocab_size):
    '''
    This function should sample HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    You\'ll need to generate HOG features using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

    However, the documentation is a bit confusing, so we will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let\'s say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a really nifty numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).

    The number of feature vectors that come from this reshape is dependent on
    the size of the image you give to hog(). It will fit as many blocks as it
    can on the image. You can choose to resize (or crop) each image to a consistent size
    (therefore creating the same number of feature vectors per image), or you
    can find feature vectors in the original sized image.

    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    to do this. Note that this can take a VERY LONG TIME to complete (upwards
    of ten minutes for large numbers of features and large max_iter), so set
    the max_iter argument to something low (we used 100) and be patient. You
    may also find success setting the \"tol\" argument (see documentation for
    details)

    Also, you may use other feature extractor like SIFT. It\'s okay to use skimage!

    suggested func:
        skimage.hog, sklearn.cluster.MiniBatchKMeans,...
    '''

    all_features = []
    # Define HOG parameters - using values similar to a SIFT block (4x4 cells)
    # The original SIFT block has 4x4 cells, each 4x4 pixels. Total descriptor size 4x4x8=128 or 4x4x9=144
    # Let's choose parameters that result in a feature vector size per block related to the description.
    # If cells_per_block is (z,z), feature vector size is z*z*9.
    # Let's try pixels_per_cell=(8, 8) and cells_per_block=(2, 2). Feature vector size per block is 2*2*9 = 36.
    # The docstring mentions (4,4) cells_per_block being equivalent to 16 cells for SIFT.
    # Let's stick closer to the docstring's SIFT analogy: pixels_per_cell=(4, 4), cells_per_block=(4, 4).
    pixels_per_cell = (4, 4)
    cells_per_block = (4, 4)
    # The feature vector size per block will be cells_per_block[0]*cells_per_block[1]*9 = 4*4*9 = 144.
    block_feature_size = cells_per_block[0] * cells_per_block[1] * 9

    print("Building vocabulary by extracting HOG features...")
    for image_path in image_paths:
        try:
            # Open image and convert to grayscale
            img = Image.open(image_path).convert('L')

            # Extract HOG features. Set feature_vector=True to get one long array.
            # We need to handle potential errors during HOG calculation, e.g., for very small images.
            try:
                hog_features_flat = hog(img, pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block, feature_vector=True,
                                        block_norm='L2-Hys') # Use L2-Hys block normalization

                # Reshape the flat features into block feature vectors
                # The total length is num_blocks * block_feature_size
                # We need to calculate the number of blocks. hog docstring isn't super clear on how num_blocks is determined when feature_vector=True.
                # Let's assume the reshape works as described: reshape(-1, block_feature_size)
                num_blocks = hog_features_flat.shape[0] // block_feature_size
                if num_blocks > 0:
                   reshaped_features = hog_features_flat.reshape(-1, block_feature_size)
                   # Collect all block features
                   all_features.append(reshaped_features)
                else:
                    print(f"No HOG blocks extracted for image {image_path}.")

            except Exception as hog_e:
                print(f"Error extracting HOG features for {image_path}: {hog_e}")
                # Continue to the next image if HOG extraction fails

        except Exception as e:
            print(f"Error processing image {image_path} for vocabulary building: {e}")
            # Continue to the next image

    # Concatenate features from all images
    if len(all_features) == 0:
        print("No features extracted for vocabulary building.")
        return np.array([])

    all_features_np = np.concatenate(all_features, axis=0)
    print(f"Extracted {all_features_np.shape[0]} local features.")

    print(f"Clustering features with MiniBatchKMeans (vocab_size={vocab_size})...")
    # Cluster the features to create the visual vocabulary
    # Set max_iter and tol for potentially faster convergence as suggested
    # Use n_init explicitly to suppress warning in newer sklearn versions
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=0, max_iter=100, tol=0.01, n_init=3)

    # Check if there are enough features to cluster
    if all_features_np.shape[0] < vocab_size:
        print(f"Warning: Number of extracted features ({all_features_np.shape[0]}) is less than vocabulary size ({vocab_size}). Adjusting vocab_size to number of features.")
        if all_features_np.shape[0] == 0:
             return np.array([]) # Cannot cluster if no features
        kmeans = MiniBatchKMeans(n_clusters=all_features_np.shape[0], random_state=0, max_iter=100, tol=0.01, n_init=3)

    kmeans.fit(all_features_np)

    print("Vocabulary built.")
    return kmeans.cluster_centers_

def get_bags_of_words(image_paths):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    '''

    # vocab is loaded before this function is called in the main script.
    # Assuming vocab is available in the scope or passed as an argument if needed,
    # based on the original structure loading 'vocab.npy'.
    # If running this function independently, ensure vocab is loaded first.
    try:
        vocab = np.load('vocab.npy')
        print('Loaded vocab from file.')
    except FileNotFoundError:
        print("Error: vocab.npy not found. Please run vocabulary building first.")
        return np.array([])

    num_images = len(image_paths)
    vocab_size = vocab.shape[0]
    # Initialize the bag of words histograms array
    bag_of_words_features = np.zeros((num_images, vocab_size), dtype=np.float32)

    # Define the same HOG parameters as used in build_vocabulary
    pixels_per_cell = (4, 4)
    cells_per_block = (4, 4)
    block_feature_size = cells_per_block[0] * cells_per_block[1] * 9

    print("Generating Bag of Words features...")
    for i, image_path in enumerate(image_paths):
        try:
            # Open image and convert to grayscale
            img = Image.open(image_path).convert('L')

            # Extract HOG features using the same settings as build_vocabulary
            try:
                hog_features_flat = hog(img, pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block, feature_vector=True,
                                        block_norm='L2-Hys')

                # Reshape the flat features into block feature vectors
                num_blocks = hog_features_flat.shape[0] // block_feature_size
                if num_blocks > 0:
                    image_features = hog_features_flat.reshape(-1, block_feature_size)

                    # For each feature in the image, find the closest visual word in the vocabulary
                    # Calculate distances from image features to vocabulary centers
                    distances = cdist(image_features, vocab, 'euclidean')

                    # Find the index of the closest vocabulary word for each image feature
                    closest_vocab_indices = np.argmin(distances, axis=1)

                    # Build the histogram by counting occurrences of each vocabulary word
                    # Use np.bincount for efficient histogram building
                    # Ensure vocab_size is correctly handled by setting minlength
                    histogram = np.bincount(closest_vocab_indices, minlength=vocab_size)

                    # Store the histogram. Ensure histogram has exactly vocab_size bins.
                    bag_of_words_features[i, :] = histogram[:vocab_size]

                else:
                     print(f"No HOG blocks extracted for image {image_path} for BoW.")

            except Exception as hog_e:
                 print(f"Error extracting HOG features for {image_path} in BoW: {hog_e}")
                 # The corresponding histogram row will remain zeros

        except Exception as e:
            print(f"Error processing image {image_path} for BoW features: {e}")
            # The corresponding histogram row will remain zeros

    # Optional: Normalize the histograms
    # Normalize each row (histogram) independently
    # sums = np.sum(bag_of_words_features, axis=1, keepdims=True)
    # bag_of_words_features = bag_of_words_features / (sums + 1e-6)

    print("Bag of Words features generated.")
    return bag_of_words_features

def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    using those learned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    We suggest you look at the sklearn.svm module, including the LinearSVC
    class. With the right arguments, you can get a 15-class SVM as described
    above in just one call! Be sure to read the documentation carefully.

    suggested function: sklearn.svm.LinearSVC
    '''

    print("Training Linear SVM classifier...")
    # Initialize the LinearSVC classifier. Use default settings or tune as needed.
    # The multiclass strategy is 'ovr' (one-vs-rest) by default for LinearSVC, which is suitable.
    # Set max_iter to a reasonable value to ensure convergence for larger datasets.
    # The docstring mentions 15 classes, adjust max_iter if needed.
    svm_classifier = LinearSVC(random_state=0, tol=1e-3, max_iter=2000) # Increased max_iter

    # Train the classifier
    svm_classifier.fit(train_image_feats, train_labels)
    print("SVM classifier trained.")

    print("Predicting test image labels with SVM...")
    # Predict the labels for the test images
    predicted_labels = svm_classifier.predict(test_image_feats)
    print("SVM predictions made.")

    return predicted_labels

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict the category for every test image by finding
    the training image with most similar features. You will complete the given
    partial implementation of k-nearest-neighbors such that for any arbitrary
    k, your algorithm finds the closest k neighbors and then votes among them
    to find the most common category and returns that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    The simplest implementation of k-nearest-neighbors gives an even vote to
    all k neighbors found - that is, each neighbor in category A counts as one
    vote for category A, and the result returned is equivalent to finding the
    mode of the categories of the k nearest neighbors. A more advanced version
    uses weighted votes where closer matches matter more strongly than far ones.
    This is not required, but may increase performance.

    Be aware that increasing k does not always improve performance - even
    values of k may require tie-breaking which could cause the classifier to
    arbitrarily pick the wrong class in the case of an even split in votes.
    Additionally, past a certain threshold the classifier is considering so
    many neighbors that it may expand beyond the local area of logical matches
    and get so many garbage votes from a different category that it mislabels
    the data. Play around with a few values and see what changes.

    Useful functions:
        scipy.spatial.distance.cdist, np.argsort, scipy.stats.mode
    '''

    k = 1

    # Gets the distance between each test image feature and each train image feature
    # e.g., cdist
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')

    # Get the indices of the k nearest neighbors for each test image
    # np.argsort sorts along the last axis by default, which is what we want (distances to training images)
    nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :k]

    # Determine the labels of the k nearest neighbors
    # train_labels is a list, convert to numpy array for easier indexing
    train_labels_np = np.array(train_labels)
    nearest_neighbor_labels = train_labels_np[nearest_neighbor_indices]

    # Find the most common label among the k neighbors for each test image
    # Use scipy.stats.mode to find the mode. It returns mode and count.
    # We are interested in the mode value, which is the first element of the first dimension.
    predicted_labels, _ = scipy.stats.mode(nearest_neighbor_labels, axis=1)

    # scipy.stats.mode can return a 2D array if there are multiple modes,
    # but for classification, we just need one prediction per test image.
    # Assuming a single prediction is expected per test sample, take the first one.
    # Reshape to a 1D array of strings as required by the output format.
    predicted_labels = predicted_labels.reshape(-1)

    return predicted_labels
