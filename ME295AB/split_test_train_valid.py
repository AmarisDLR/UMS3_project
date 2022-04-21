import os, splitfolders



base_dir = 'UMS3_dd_April10'
images_dir = os.path.join(base_dir,'all')
output_dir = os.path.join(base_dir,'training_randomized')

## ratio argument(train:val:test)
train_size = 0.85
test_size = 0.03
valid_size = 1-train_size-test_size

splitfolders.ratio(images_dir, output=output_dir, seed=1337, ratio=(train_size, valid_size, test_size ))
print('Done.')

