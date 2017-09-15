function [] = exp_texture_escher_3()

rng(123);

% config
img_name = 'escher';
patch_size = 48;

% setup
use_gpu = 1;
compile_convnet = 0;

% setup
setup_path();
setup_convnet(use_gpu, compile_convnet);

% (1)
num = 1;

Gammas = 10; % [0.1 1 10];
Deltas = [0.1 0.3 0.5 1];
Ts = [15];

Delta2 = 0.0003;
Gamma2 = 5;

for Delta = Deltas
    for Gamma = Gammas
        for T = Ts
            % prep
            prefix = [img_name '/3_' num2str(num) '/'];
            [config, net1] = train_coop_config();
            config = prep_images(config, ['../data/' img_name '/'], patch_size);
            config = prep_dirs(config, prefix);
            config.im_size = patch_size;
            config.file_str = [img_name '/'];
            config.use_gpu = use_gpu;
            config.nIteration = 200;   
            config.batchSize = 50;

            % sampling parameters
            config.num_syn = 32;
            
            % descriptor net1 parameters
            config.Delta = Delta;
            config.Gamma = [0.0005*ones(1,100), 0.00005*ones(1,100), 0.00001*ones(1,100), 0.000005*ones(1,100), 0.000001*ones(1,100)] * Gamma;
            config.refsig = 1;                                            
            config.T = T;            
            
            % generator net2 parameters
            config.Delta2 = Delta2;
            config.Gamma2 = [0.0002*ones(1,100), 0.0001*ones(1,100), 0.00008*ones(1,100), 0.00006*ones(1,100), 0.00004*ones(1,100)] * Gamma2;
            config.refsig2 = 1;
            config.s = 0.3;
            config.real_ref = 1;
            config.cap2 = 8;

            % learn
            learn_dual_net(config, net1);

            num = num + 1;
        end
    end
end



end

function [config] = prep_images(config, patch_path, patch_size)
[mean_im, imdb] = load_images(patch_path, patch_size);
config.mean_im = mean_im;
config.imdb = imdb;
end

function [config] = prep_dirs(config, prefix)
config.trained_folder = [config.trained_folder prefix];
config.gen_im_folder = [config.gen_im_folder prefix];
config.syn_im_folder = [config.syn_im_folder prefix];
if ~exist(config.trained_folder,'dir') mkdir(config.trained_folder); end
if ~exist(config.gen_im_folder,'dir') mkdir(config.gen_im_folder); end
if ~exist(config.syn_im_folder,'dir') mkdir(config.syn_im_folder); end
end

function [mean_im, imdb] = load_images(img_path, img_size)
files = dir([img_path, '*.png']);
imdb = zeros(img_size, img_size,3,length(files));
for i = 1:length(files)
    imdb(:,:,:,i) = imread([img_path,files(i).name]);
end
mean_im = single(128*ones(img_size,img_size,3));
imdb = single(imdb - repmat(mean_im,1,1,1,size(imdb,4)));
end
