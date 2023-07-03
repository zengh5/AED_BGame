% Mount test2, and combine its results to get a M byN PD matrix: 
% M: the number of strength;
% N: number of Pfa1, 0 : 0.01: P_fa. Let's say Pfa=0.05, then N = 6;
clear,clc
addpath('Test2')

% 1 Load the pre-trained ensemble classifier
load Test2/model_51_ori_mixed_PGDresave.mat

% 2 Load the threshold as a function of P_fa2, learned from a number of
% benign images
load Test2/Threshold_fpr2PGDresave.mat

% 3 read the image ans extract SRM features
strength_list = {'PGD01', 'PGD02', 'PGD04'};
img_list = {'10.png', '12.png', '13.png', '14.png', '15.png',...
            '18.png', '19.png', '20.png', '21.png', '22.png'};
for i_P_fa1 = 0:5
    %  P_fa2 = P_fa - P_fa1;
    P_fa2 = 5 - i_P_fa1;
    if i_P_fa1 == 0
        threshold = (Threshold_fpr2(P_fa2)+Threshold_fpr2(P_fa2+1))/2;
        for i_strength = 1:3
            strength = strength_list{i_strength};
            escape_ind2 = zeros(1, 10);
            for i_img = 1:10
                imagepath = ['images/', strength '/', img_list{i_img}];
                fea = SRM34671(imagepath);
                % 4 Classify the features with a pre-trained ensemble classifier
                test_results = ensemble_testing(fea,trained_ensemble);

                %% test_results: the classification results of N sub classifiers
                %% We do not use the 'prediction' variable directly as that done in steganalysis
                if test_results.votes <= threshold
                    disp('The image is benign'),
                    escape_ind2(i_img) = 1;
                else
                    disp('The image is adversarial'),
                end
            end
            escape_two_step = escape_ind2;
            PD(i_strength, i_P_fa1+1) = (10-sum(escape_two_step))/10;
            done =1;
        end

    else
        if P_fa2 == 0
            threshold = 100; % A sufficient large number
        else
            threshold = (Threshold_fpr2(P_fa2)+Threshold_fpr2(P_fa2+1))/2;
        end

        for i_strength = 1:3
            strength = strength_list{i_strength};
            escape_ind2 = zeros(1, 10);
            for i_img = 1:10
                imagepath = ['images/', strength '/', img_list{i_img}];
                fea = SRM34671(imagepath);
                % 4 Classify the features with a pre-trained ensemble classifier
                test_results = ensemble_testing(fea,trained_ensemble);

                %% test_results: the classification results of N sub classifiers
                %% We do not use the 'prediction' variable directly as that done in steganalysis
                if test_results.votes <= threshold
                    disp('The image is benign'),
                    escape_ind2(i_img) = 1;
                else
                    disp('The image is adversarial'),
                end
            end
            matname = ['mat1/' num2str(i_P_fa1) '_' strength_list{i_strength} '.mat'];
            load(matname)
            escape_two_step = escape_ind2 & escape_ind1;
            PD(i_strength, i_P_fa1+1) = (10-sum(escape_two_step))/10;
        end
    end
end
matname = 'PD_Pfa05.mat';
save(matname, 'PD');
Done = 1;









