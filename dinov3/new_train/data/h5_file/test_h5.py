import h5py


# file_path = '/mnt/unicom_nas/data/inhouse/ssl/bundles/6a808ffd7868ce959c2894edb949ac2c/2023-124548#2#1.h5'
file_path = '/mnt/local10/data_new/tcga_rundata/patches_output/patches/TCGA-2A-A8VL-01A-02-TS2.AFBBB2D5-39E6-434A-B6E5-779DD8217DCD.h5'
with h5py.File(file_path, "r") as f:
    print(f.keys())
    d = f["coords"]
    print("shape:", d.shape)
    print("chunks:", d.chunks)
    print("compression:", d.compression)




