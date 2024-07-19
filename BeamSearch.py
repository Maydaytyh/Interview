import torch
import torch.nn.functional as F

def beam_search(LM_prob,beam_size=3):
    batch,seqlen,vocab_size = LM_prob.shape
    log_LM_prob = LM_prob.log()
    log_beam_prob, indices = log_LM_prob[:,0,:].topk(beam_size,sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1,seqlen):
        log_beam_prob = log_beam_prob.unsqueeze(-1) + log_LM_prob[:,i,:].unsqueeze(1).repeat(1,beam_size,1)
        log_beam_prob,index = log_beam_prob.view(batch,-1).topk(beam_size,sorted=True)
        beam_id = index // vocab_size
        index = index%vocab_size
        mid = torch.Tensor([])
        for j,bid,idx in zip(range(batch),beam_id,index):
            x = torch.cat([indices[j][bid],idx.unsqueeze(-1)],-1)
            mid = torch.cat([mid,x.unsqueeze(0)],0)
        indices = mid
    return indices,log_beam_prob





if __name__=='__main__':
    LM_prob = F.softmax(torch.randn([1,2,3]),dim = -1)
    print(LM_prob)
    indices,log_prob = beam_search(LM_prob,beam_size = 3)
    print(indices)
