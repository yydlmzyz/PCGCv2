
def eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, res, max_num):
    print('\n', '\n', '===============', time.asctime(time.localtime(time.time())), filedir)
    for idx_ckpt, ckptdir in enumerate(ckptdirs):
        voxel_size = voxel_sizes[idx_ckpt]
        rho = rhos[idx_ckpt]

        print('\n======', idx_ckpt, ckptdir, voxel_size)
        
        start_time = time.time()
        if os.path.exists(ckptdir):
            ckpt = torch.load(ckptdir)
            pcc.encoder.load_state_dict(ckpt['encoder'])
            pcc.decoder.load_state_dict(ckpt['decoder'])
            pcc.entropy_bottleneck.load_state_dict(ckpt['entropy_bottleneck'])
        else:
            print('load failed!')
            break
        #
        #prefix = os.path.split(filedir)[-1].split('.')[0] + '_R' + str(idx_ckpt)
        prefix = os.path.split(filedir)[-1].split('.')[0]
        
        #coords_binname, feats_binname, head_binname = encode(filedir, pcc, prefix)
        partition_start = time.time()
        partition_filedirs, partition_num = partition_point_cloud(filedir, max_num)
        torch.cuda.synchronize()
        partition_time = round(time.time() - partition_start, 6)
        
        encode_start = time.time()
        coords_binname, feats_binname, head_binname = partition_encode(partition_filedirs, pcc, max_num, voxel_size)
        all_binname = write_file(prefix, len(partition_filedirs))
        torch.cuda.synchronize()
        encode_time = round(time.time() - encode_start, 6)
        
        decode_start = time.time()
        coords_binname, feats_binname, head_binname = load_file(prefix)
        #out = decode(coords_binname, feats_binname, head_binname, pcc, rho)
        out = partition_decode(coords_binname, feats_binname, head_binname, pcc, rho=rho, voxel_size=voxel_size)
        torch.cuda.synchronize()
        decode_time = round(time.time() - decode_start, 6)
        
        metric_start = time.time()
        results = metrics(filedir, out, coords_binname, feats_binname, head_binname, all_binname, res)
        torch.cuda.synchronize()
        metric_time = round(time.time() - metric_start, 6)

        results["voxel size"] = voxel_size
        results["rho"] = rho

        # time
        results["partition time"] = partition_time
        results["partition num"] = partition_num
        results["encode time"] = encode_time
        results["decode time"] = decode_time
        results["metric time"] = metric_time
        results["time"] = round(time.time() - start_time, 6)
        print('partition num:', partition_num, '\n',
              'PSNR (D1/D2):', results["mseF,PSNR (p2point)"][0],  results["mseF,PSNR (p2plane)"][0], '\n',
              'bpp:', results["bpp"][0], results["bpp_feats"][0], results["bpp_coords"][0], '\n',
              'time:', results["time"][0], results["encode time"][0], results["decode time"][0])
        if idx_ckpt == 0:
            all_results = results.copy(deep=True)
        else:
            all_results = all_results.append(results, ignore_index=True)

    csv_name = os.path.join(csv_root_dir, os.path.split(filedir)[-1].split('.')[0] + '.csv')
    all_results.to_csv(csv_name, index=False)

    # plot
    def plot(csv_name):
        all_results = pd.read_csv(csv_name)
        fig, ax = plt.subplots(figsize=(7.3, 4.2))
        plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
                label="D1", marker='x', color='red')
        plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
                label="D2", marker='x', color='blue')
        filename = os.path.split(csv_name)[-1][:-4]
        plt.title(filename)
        plt.xlabel('bpp')
        plt.ylabel('RSNR')
        plt.grid(ls='-.')
        plt.legend(loc='lower right')
        fig.savefig(csv_name[:-4]+'.png')
        return 
    plot(csv_name)

# ## run
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", type=str, default='testdata/8iVFB/longdress_vox10_1300_n.ply', help="filedir")
    parser.add_argument("--csvrootdir", type=str, default='results/mulriscalepcgc', help="csvrootdir")
    parser.add_argument("--res", type=int, default=1024, help="resolution")
    parser.add_argument("--max_num", type=int, default=1e6, help="max number of points")
    parser.add_argument("--rho", type=float, default=1, help="output_num/input_num")
    parser.add_argument("--test_all", default=False, action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    from PCCModel import PCC
    pcc = PCC(channels=8).to(device)
    print("total params:", sum([param.nelement() for param in pcc.parameters()]))
    
    args = parse_args()
    filedir = args.filedir
    csv_root_dir = args.csvrootdir
    if not os.path.exists(csv_root_dir):
        os.makedirs(csv_root_dir)
    res = args.res
    max_num = args.max_num

    if args.test_all:
        ckptdirs = [
            './ckpts/c8_a025_14000.pth', 
            './ckpts/c8_a05_32000.pth', 
            './ckpts/c8_a1_32000.pth', 
            './ckpts/c8_a2_32000.pth',  
            './ckpts/c8_a4_32000.pth', 
            './ckpts/c8_a6_32000.pth', 
            './ckpts/c8_a10_32000.pth']

        voxel_sizes = [1, 1, 1, 1, 1, 1, 1]
        # 8i
        rhos = [1.4, 1.2, 1, 1, 1, 1, 1]
        # rhos = [1, 1, 1, 1, 1, 1, 1] # for D2
        start_eval = time.time()
        filedir = 'testdata/8iVFB/longdress_vox10_1300_n.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 1024, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')

        start_eval = time.time()
        filedir = 'testdata/8iVFB/redandblack_vox10_1550_n.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 1024, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')
        
        start_eval = time.time()
        filedir = 'testdata/8iVFB/loot_vox10_1200_n.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 1024, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')

        start_eval = time.time()
        filedir = 'testdata/8iVFB/soldier_vox10_0690_n.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 1024, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')

        # mvub
        rhos = [1.3, 1.2, 1, 1, 1, 1, 1]
        # rhos = [1, 1, 1, 1, 1, 1, 1] # for D2
        start_eval = time.time()
        filedir = 'testdata/MVUB/andrew_vox9_frame0000.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 512, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')

        start_eval = time.time()
        filedir = 'testdata/MVUB/david_vox9_frame0000.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 512, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')
        
        start_eval = time.time()
        filedir = 'testdata/MVUB/phil_vox9_frame0139.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 512, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')

        start_eval = time.time()
        filedir = 'testdata/MVUB/sarah_vox9_frame0023.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 512, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')

        # owlii
        rhos = [1.2, 1.1, 1, 1, 1, 1, 1]
        # rhos = [1, 1, 1, 1, 1, 1, 1] # for D2
        start_eval = time.time()
        filedir = 'testdata/Owlii/basketball_player_vox11_00000200.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 2048, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')

        start_eval = time.time()
        filedir = 'testdata/Owlii/dancer_vox11_00000001.ply'
        eval(filedir, csv_root_dir, ckptdirs, voxel_sizes, rhos, 2048, max_num)
        print('======time:', round(time.time() - start_eval, 4))
        os.system('rm *.ply *.bin')
        
