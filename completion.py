""" This script attempts at performing data completion
(i.e. predicting a new contrast given previous contrasts)
based on the localizer public dataset.

Author: Bertrand Thirion, 2014
"""

import numpy as np
from os import mkdir, getcwd, path as op
import warnings
import pickle


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoLarsCV
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cross_validation import ShuffleSplit
from sklearn.pls import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from joblib import Parallel, delayed
from nibabel import load, save, Nifti1Image
from nilearn.input_data import NiftiMasker
from nilearn import datasets
from nilearn.masking import compute_multi_background_mask, intersect_masks


###############################################################################
# Get the data

# note : the following contrasts correspond to the raw conditions
# of the localizer experiment
contrasts = ["horizontal checkerboard",
             "vertical checkerboard",
             'sentence listening', "sentence reading",
             "calculation (auditory cue)",
             "calculation (visual cue)",
             "left button press (auditory cue)",
             "left button press (visual cue)",
             "right button press (auditory cue)",
             "right button press (visual cue)"]

test_set = ['left button press (auditory cue)']
ref = [contrast for contrast in contrasts if contrast not in test_set]
n_ref = len(ref)

nifti_masker = NiftiMasker('mask_GM_forFunc.nii')
affine = load(nifti_masker.mask).get_affine()

# fetch the data
ref_imgs = datasets.fetch_localizer_contrasts(ref).cmaps
n_subjects = len(ref_imgs) / n_ref

# Create a population mask
one_contrast = [img for img in ref_imgs if 'horizontal' in img]
mask_ = compute_multi_background_mask(one_contrast)
mask_image = intersect_masks(['mask_GM_forFunc.nii', mask_])
mask = mask_image.get_data()
n_voxels = mask.sum()
save(mask_image, '/tmp/mask.nii')
nifti_masker = NiftiMasker(mask_image)

# write directory
write_dir = op.join(getcwd(), 'results')
if not op.exists(write_dir):
    mkdir(write_dir)


###############################################################################

# Global parameters
n_clusters = 5000

test_set = ['left button press (auditory cue)']
do_soft_threshold = False
nifti_masker = NiftiMasker(mask=mask_image, smoothing_fwhm=False,
                           standardize=False, memory='nilearn_cache')
shape = mask.shape
connectivity = grid_to_graph(n_x=shape[0], n_y=shape[1], n_z=shape[2],
                             mask=mask)

#cross_validation scheme
subject_label = np.repeat(np.arange(n_subjects), len(ref))
cv = ShuffleSplit(n_subjects, n_iter=20, train_size=.9,
                  test_size=.1, random_state=2)


def do_parcel_connectivity(mask, n_clusters, ward):
    # Estimate parcel connectivity
    import scipy.sparse as sps
    n_voxels = mask.sum()
    incidence_ = np.zeros((n_voxels, n_clusters))
    incidence_[np.arange(n_voxels), ward.labels_] = 1
    incidence = sps.coo_matrix(incidence_)
    parcel_connectivity = (
        (incidence.T * connectivity) * incidence).toarray() > 0
    return parcel_connectivity


def prepare_data(imgs, connectivity, mask, n_clusters=5000, n_components=100):
    # data preparation
    Z = nifti_masker.fit_transform(imgs)
    pca = RandomizedPCA(n_components=n_components)
    Z_ = pca.fit_transform(Z.T).T
    ward = WardAgglomeration(n_clusters=n_clusters, connectivity=connectivity,
                             memory='nilearn_cache').fit(Z_)
    W = ward.transform(Z)
    del Z
    # data cube is a more convenient representation
    cube = np.array([W[subject_label == subject]
                     for subject in np.arange(n_subjects)])
    # parcel connectivity
    parcel_connectivity = do_parcel_connectivity(mask, n_clusters, ward)
    return cube, ward, parcel_connectivity


def simplest(cube, y, cv):
    """ just use the mean to impute the missing values
    """
    from sklearn.dummy import DummyRegressor
    clf = DummyRegressor()
    X = cube.reshape(cube.shape[0], cube.shape[1] * cube.shape[2])
    sse = np.zeros(y.shape[1])
    for train, test in cv:
        y_train, y_test = y[train], y[test]
        y_predict = clf.fit(X[train], y[train]).predict(X[test])
        sse += np.mean((y_predict - y_test) ** 2, 0)
    return sse


def plss(X, y, cv, n_components=1):
    """
    """
    pls = PLSRegression(n_components=n_components)
    sse = np.zeros(y.shape[1])
    for train, test in cv:
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        y0 = y_train.mean(0)
        X0 = X_train.mean(0)
        pls.fit(X_train - X0, y_train - y0)
        sse += np.sum((y_test - y0 - pls.predict(X_test - X0)) ** 2, 0)
    return sse


def scorer2(x, y, pen, n_components, clf, cv, tuned_parameters):
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    n_samples = x.shape[0] * .8
    if x.shape[1] > n_samples:
        select = np.arange(x.shape[1])
        np.random.shuffle(select)
        select = select[:n_samples]
        x = x[:, select]

    if pen in ['ridge', 'lasso'] or len(n_components) == 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sse = -cross_val_score(clf, x, y, cv=cv, n_jobs=1,
                                      scoring='mean_squared_error').sum()
    else:
        sse = 0
        for train, test in cv:
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]
            pclf = GridSearchCV(clf, tuned_parameters, cv=5,
                                scoring='mean_squared_error', n_jobs=1)
            pclf.fit(x_train, y_train)
            sse += np.mean((y_test - pclf.predict(x_test)) ** 2)
    return sse


def low_rank_regional(cube, y, cv, p_connectivity, n_components=[4],
                      fit_intercept=True, pen='rank'):
    """
    penalty: {'rank', 'ridge', 'lasso'} string
    """
    tuned_parameters = []
    if pen == 'rank':
        lr = LinearRegression(fit_intercept=fit_intercept)
        pca = PCA(n_components=n_components[0])
        clf = Pipeline([('pca', pca), ('reg', lr)])
        tuned_parameters = [{'pca__n_components': n_components}]
    elif pen == 'ridge':
        clf = RidgeCV(fit_intercept=fit_intercept)
    elif pen == 'lasso':
        clf = LassoLarsCV(fit_intercept=fit_intercept)
    elif pen == 'trees':
        clf = ExtraTreesRegressor(n_estimators=10, max_features='auto',
                                       random_state=0)
    elif pen == 'knn':
        clf = KNeighborsRegressor()
    else:
        clf = LinearRegression(fit_intercept=fit_intercept)

    sse = Parallel(n_jobs=1)(delayed(scorer2)(
            cube.T[p_connectivity[i]].T, Y.T[i], pen, n_components, clf,
            cv, tuned_parameters)
                              for i in range(len(Y.T)))

    return np.array(sse)


def scorer(xT, y, pen, n_components, clf, cv, tuned_parameters):
    x = xT.T
    if pen in ['ridge', 'lasso'] or len(n_components) == 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sse = -cross_val_score(clf, x, y, cv=cv, n_jobs=1,
                                      scoring='mean_squared_error').sum()
    else:
        sse = 0
        for train, test in cv:
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]
            pclf = GridSearchCV(clf, tuned_parameters, cv=5,
                                scoring='mean_squared_error', n_jobs=1)
            pclf.fit(x_train, y_train)
            sse += np.mean((y_test - pclf.predict(x_test)) ** 2)
    return sse


def low_rank_local(cube, y, cv, n_components=[4], fit_intercept=True,
                   pen='rank'):
    """
    penalty: {'rank', 'ridge', 'lasso', 'tls'} string
    """
    tuned_parameters = []
    if pen == 'rank':
        lr = LinearRegression(fit_intercept=fit_intercept)
        pca = PCA(n_components=n_components[0])
        clf = Pipeline([('pca', pca), ('reg', lr)])
        tuned_parameters = [{'pca__n_components': n_components}]
    elif pen == 'ridge':
        clf = RidgeCV(fit_intercept=fit_intercept)
    elif pen == 'lasso':
        clf = LassoLarsCV(fit_intercept=fit_intercept)
    elif pen == 'trees':
        clf = ExtraTreesRegressor()
    elif pen == 'knn':
        clf = KNeighborsRegressor()
    else:
        clf = LinearRegression(fit_intercept=fit_intercept)

    sse = Parallel(n_jobs=1)(delayed(scorer)(xT, y, pen, n_components, clf,
                                              cv, tuned_parameters)
                              for (xT, y) in zip(cube.T, Y.T))

    return np.array(sse)


def low_rank_pca(cube, Y, cv, n_components=400, reg_rank=[6],
                 fit_intercept=True, pen='rank'):
    """ PCR on spatial features matrix"""
    if pen == 'rank':
        clf = LinearRegression(fit_intercept=True)
        pca_ = PCA(n_components=reg_rank[0])
        tuned_parameters = [{'pca__n_components': reg_rank}]
        clf = Pipeline([('pca', pca_), ('reg', clf)])
    elif pen == 'ridge':
        clf = RidgeCV(fit_intercept=fit_intercept)
    elif pen == 'lasso':
            clf = LassoLarsCV(fit_intercept=fit_intercept)
    elif pen == 'trees':
        clf = ExtraTreesRegressor(n_estimators=10, max_features='auto',
                                       random_state=0)
    elif pen == 'knn':
        clf = KNeighborsRegressor()
    else:
        clf = LinearRegression(fit_intercept=fit_intercept)
    pca = PCA(n_components=n_components)
    W = cube.T.reshape(n_clusters, n_subjects * n_ref).T
    w = pca.fit_transform(W)
    sse = np.zeros(Y.shape[1])
    for train, test in cv:
        Y0 = Y[train].mean(0)
        y = pca.transform(Y[train] - Y0)
        proj = []
        for x, y_train in zip(w.T, y.T):
            x_ = x.reshape(n_ref, n_subjects).T
            x_train, x_test = x_[train], x_[test]
            if pen in ['ridge', 'lasso'] or len(reg_rank) == 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(x_train, y_train)
                    proj.append(clf.predict(x_test))
            else:
                pclf = GridSearchCV(clf, tuned_parameters, cv=5,
                                    scoring='mean_squared_error', n_jobs=1)
                pclf.fit(x_train, y_train)
                proj.append(pclf.predict(x_test))
        proj = np.array(proj).T
        Y_pred = pca.inverse_transform(proj) + Y0
        sse += np.mean((Y[test] - Y_pred) ** 2, 0)
    return sse


cube, ward, parcel_connectivity = prepare_data(
    ref_imgs, connectivity, mask, n_clusters=n_clusters)


if 1:
    # run the actual experiment: compute scores using various methods
    results = {}
    for test_contrast in test_set:
        results_ = {}
        test_imgs = datasets.fetch_localizer_contrasts([test_contrast]).cmaps
        Y = ward.transform(nifti_masker.fit_transform(test_imgs))
        print test_contrast
        results_['dummy'] = simplest(cube, Y, cv)
        results_['pcr_local'] = low_rank_local(cube, Y, cv)
        results_['knn_local'] = low_rank_local(cube, Y, cv, pen='knn')
        results_['trees_local'] = low_rank_local(cube, Y, cv, pen='trees')
        results_['ridge_local'] = low_rank_local(
            cube, Y, cv, fit_intercept=True, pen='ridge')
        results_['lasso_local'] = low_rank_local(
            cube, Y, cv, fit_intercept=True, pen='lasso')
        results_['pcr_local_cv'] = low_rank_local(
            cube, Y, cv, fit_intercept=True, n_components=np.arange(1, 6))
        results_['pcr_regional'] = low_rank_regional(
            cube, Y, cv, parcel_connectivity, n_components=[5])
        results_['knn_regional'] = low_rank_regional(
            cube, Y, cv, parcel_connectivity, pen='knn')
        results_['trees_regional'] = low_rank_regional(
            cube, Y, cv, parcel_connectivity, pen='trees')
        results_['lasso_regional'] = low_rank_regional(
            cube, Y, cv, parcel_connectivity, pen='lasso')
        results_['ridge_regional'] = low_rank_regional(
            cube, Y, cv, parcel_connectivity, pen='ridge')
        results_['pcr_global'] = low_rank_pca(cube, Y, cv)
        results_['pcr_global_cv'] = low_rank_pca(
            cube, Y, cv, reg_rank=np.arange(1, 10))
        results_['ridge_global'] = low_rank_pca(cube, Y, cv, pen='ridge')
        results_['lasso_global'] = low_rank_pca(cube, Y, cv, pen='lasso')
        results_['knn_global'] = low_rank_pca(cube, Y, cv, pen='knn')
        results_['trees_global'] = low_rank_pca(cube, Y, cv, pen='trees')
        for (key, values) in results_.iteritems():
            print key, values.sum()
        results[test_contrast] = results_

    fid = open(op.join(write_dir, 'results_%s.pickle' % test_set, 'w'))
    pickle.dump(results, fid)
    fid.close()
else:
    # do an exploratory analysis
    ratios = []
    stats = []
    for test_contrast in test_set:
        test_imgs = datasets.fetch_localizer_contrasts([test_contrast]).cmaps
        Y = ward.transform(nifti_masker.fit_transform(test_imgs))
        print test_contrast
        sim = simplest(cube, Y, cv)
        print sim.sum()
        res = low_rank_local(cube, Y, cv, fit_intercept=True)
        print res.sum(), res.sum() / sim.sum()
        ratio = 1 - res / sim
        stat = Y.mean(0)

        ratio_img = nifti_masker.inverse_transform(
            ward.inverse_transform(ratio))
        save(ratio_img, op.join(write_dir, 'ratio_%s.nii' % test_contrast))
        mean_img = nifti_masker.inverse_transform(ward.inverse_transform(stat))
        save(mean_img, op.join(write_dir, 'stat_%s.nii' % test_contrast))
        stats.append(stat)
        ratios.append(ratio)

    ratios = np.array(ratios)
    stats = np.array(stats)
    stat = np.sqrt(np.sum(stats ** 2, 0))
    ratio = np.mean(ratios, 0)
    save(nifti_masker.inverse_transform(ward.inverse_transform(ratio)),
         op.join(write_dir, 'ratio.nii'))
    save(nifti_masker.inverse_transform(ward.inverse_transform(stat)),
         op.join(write_dir, 'stat.nii'))
