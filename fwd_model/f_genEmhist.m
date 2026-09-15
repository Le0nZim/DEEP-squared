function emhist = f_genEmhist(max_input_photons,N_reps,pram)
% Conditional EM table with verifiable gain parameters. Honors the caller RNG.
% Supports every positive trial count, including <100 and nonmultiples of 100.
validateattributes(max_input_photons,{'numeric'},{'scalar','integer','positive'});
validateattributes(N_reps,{'numeric'},{'scalar','integer','positive'});
validateattributes(pram.cam_N_gainStages,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(pram.cam_Brnuli_alpha,{'numeric'},{'scalar','>=',0,'<=',1});
values=zeros(max_input_photons,N_reps);
for first=1:100:N_reps
    last=min(first+99,N_reps);
    block=repmat((1:max_input_photons)',1,last-first+1);
    for stage=1:pram.cam_N_gainStages
        block=block+binornd(block,pram.cam_Brnuli_alpha);
    end
    values(:,first:last)=block;
end
emhist=struct('values',values,'cam_N_gainStages',pram.cam_N_gainStages,...
    'cam_Brnuli_alpha',pram.cam_Brnuli_alpha);
pram_usedInEmhist=pram;
folder=fullfile('.','_emhist');
if isfield(pram,'emhist_dir'), folder=pram.emhist_dir; end
if ~isfolder(folder), mkdir(folder); end
save(fullfile(folder,['emhist_' pram.dataset '_' date '.mat']),'emhist','pram_usedInEmhist');
end
