import h5py

with h5py.File("/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/training_data/grab_roller/demo_grab_roller/grab_roller-demo_clean-50/episode_0/episode_0.hdf5", "r") as f:
    def print_all_keys(h5obj, prefix=""):
        for key in h5obj.keys():
            item = h5obj[key]
            if isinstance(item, h5py.Group):
                print(f"{prefix}{key}/")
                print_all_keys(item, prefix + key + "/")
            else:
                print(f"{prefix}{key}  {item.shape}")

    print_all_keys(f)
