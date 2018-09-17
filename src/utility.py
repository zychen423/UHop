import torch
import os

def log_result(num, ques, relas, rela_texts, scores, acc, path, word2id, rela2id):
    id2word = {v: k for k, v in word2id.items()}
    id2rela = {v: k for k, v in rela2id.items()}
    with open(os.path.join(path, 'error.log'), 'a') as f:
        f.write(f'\n{num} ==============================\n')
        q = [id2word[x] for x in ques[0].data.cpu().numpy()]
        f.write(' '.join(q)+'\n') 
        f.write('Correct:\n')
        t = [id2word[x] for x in rela_texts[0].data.cpu().numpy()]
        f.write(' '.join(t)+'\n') 
        r = [id2rela[x] for x in relas[0].data.cpu().numpy()]
        f.write(' '.join(r)+'\n') 
        c_s = scores[0]
        f.write(str(c_s.data.cpu().numpy())+'\n') 
        if acc == 1:
            f.write('Result:Correct!\n')
        else:
            f.write('Result:Incorrect! ====================\n\n')
            for q, r, t, s in zip(ques[1:], relas[1:], rela_texts[1:], scores[1:]):
                if s > c_s:
                    t = [id2word[x] for x in t.data.cpu().numpy()]
                    f.write(' '.join(t)+'\n') 
                    r = [id2rela[x] for x in r.data.cpu().numpy()]
                    f.write(' '.join(r)+'\n') 
                    f.write(str(s.data.cpu().numpy())+'\n') 
        f.write('\n====================================\n')

def find_save_dir(parent_dir, model_name):
    counter = 0
    save_dir = f'../{parent_dir}/{model_name}_{counter}'
    while os.path.exists(save_dir):
        counter += 1
        save_dir = f'../{parent_dir}/{model_name}_{counter}'
    os.mkdir(save_dir)
    print(f'save_dir is {save_dir}')
    return save_dir

def save_model(model, path):
    path = os.path.join(path, 'model.pth')
    torch.save(model.state_dict(), path)
    print(f'save model at {path}')

def save_model_with_result(model, loss, acc, rc, td, td_rc, path):
    path = os.path.join(path, f'model_{loss:.4f}_{acc:.4f}_{rc:.2f}_{td:.2f}_{td_rc:.2f}.pth')
    torch.save(model.state_dict(), path)
    print(f'save model at {path}')

def load_model(model, path):
    path = os.path.join(path, 'model.pth')
    print(f'load model from: {path}')
    model.load_state_dict(torch.load(path))
    return model


