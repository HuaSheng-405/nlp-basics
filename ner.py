import json
import torch 
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForTokenClassification,AutoTokenizer,get_scheduler
from datasets import load_dataset,Dataset as HFDataset
from sklearn.metrics import classification_report,f1_score
from tqdm import tqdm


def align_labels(tokenizer,max_length,label_to_id):
    def inner_fn(examples):
        """把字符级实体标注对齐到token级BIO标注"""
        tokenized_inputs = tokenizer(
            examples["text"],padding = "max_length",truncation = True,
            max_length = max_length,return_offsets_mapping = True
        )
        all_labels = []
        for i,text in enumerate(examples['text']):
            offset_mapping = tokenized_inputs['offset_mapping'][i]  #每个字符对应的token区间
            label_dict = examples['label'][i]       # {"name": {"叶老桂": [[9,11]]}, ...}

            #给每个字符标记BIO
            char_labels = ["O"] * len(text)
            for entity_type,entities in label_dict.items():
                for entity_name,positions in entities.items():
                    for start,end in positions:
                        char_labels[start] = f"B-{entity_type}"
                        for j in range(start + 1,end + 1):
                            char_labels[j] = f"I-{entity_type}"

            # 把字符标签映射到 token（[CLS] 和 [SEP] 标 -10O，pad 标 -100）
            token_labels = []
            for offset in offset_mapping:
                char_start,char_end = offset
                if char_start == 0 and char_end == 0:
                    # [CLS], [SEP], [PAD] 特殊 token
                    token_labels.append(-100)
                else:
                    # 取这个 token 第一个字符的标签
                    token_labels.append(label_to_id[char_labels[char_start]])
            all_labels.append(token_labels)
        tokenized_inputs["labels"] = all_labels
        del tokenized_inputs["offset_mapping"]
        return tokenized_inputs
    return inner_fn

def get_dataset(tokenizer):
    #1.加载数据
    dataset = load_dataset("json",data_files={
        "train":'D:/learn/llm/nlp-basics/data/train.json',
        "validation":'D:/learn/llm/nlp-basics/data/dev.json'
        })

    #2.查看输出类别
    all_labels = set()
    for sample in dataset['train']:
        all_labels.update(sample["label"].keys())
    label_list = sorted(all_labels)
    print(f"实体类别 ({len(label_list)}): {label_list}")

    #3.BIO标签
    bio_labels = ["O"]
    for label in label_list:
        bio_labels.append(f"B-{label}")
        bio_labels.append(f"I-{label}")

    label_to_id = {v:k for k,v in enumerate(bio_labels)}
    id_to_label = {v:k for k,v in label_to_id.items()}
    print(f"BIO 标签数: {len(bio_labels)}")
    print(bio_labels)

    #4.数据预处理
    max_length = 128

    train_dataset = dataset['train'].map(align_labels(tokenizer,max_length,label_to_id),batched = True)
    val_dataset = dataset['validation'].map(align_labels(tokenizer,max_length,label_to_id),batched = True)

    train_dataset.set_format(type = "torch",columns = ["input_ids","attention_mask","labels"])
    val_dataset.set_format(type = "torch",columns = ["input_ids","attention_mask","labels"])

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    return train_dataset,val_dataset,bio_labels,id_to_label,label_to_id

def train_evaluate(model,train_loader,val_loader,device):
    optimizer = torch.optim.Adam(model.parameters(),lr = 2e-5)
    epochs = 3
    training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",optimizer = optimizer,
        num_warmup_steps = int(0.1 * training_steps),
        num_training_steps = training_steps
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader,desc = f"Epoch{epoch + 1}训练"):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds,all_labels_list = [],[]
        for batch in tqdm(val_loader,desc = f"Epoch{epoch + 1}验证"):
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            preds = torch.argmax(outputs.logits,dim = -1)
            all_preds.extend(preds.cpu().tolist())
            all_labels_list.extend(batch["labels"].cpu().tolist())

        # 过滤掉 -100
        flat_preds,flat_labels = [],[]
        for seq_pred,seq_label in zip(all_preds,all_labels_list):
            for p,l in zip(seq_pred,seq_label):
                if l != -100:
                    flat_preds.append(p)
                    flat_labels.append(l)

        f1 = f1_score(flat_labels,flat_preds,average = "micro") * 100
        print(f"Epoch {epoch + 1}:Loss {avg_loss:.4f},F1 {f1:.2f}")
    torch.save(model.state_dict(),'D:/learn/llm/nlp-basics/model/ner.pt')
def predict_ner(text,model,tokenizer,id_to_label,device):
    inputs = tokenizer(text,return_tensors = "pt",truncation = True,max_length = 128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    token_preds = torch.argmax(outputs.logits,dim = -1)[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    current_entity = ""
    current_type = ""
    for token,pred_id in zip(tokens,token_preds):
        label = id_to_label[pred_id]
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity,current_type))
            current_entity = token
            current_type = label[2:]
        elif label.startswith("I-"):
            current_entity += token.lstrip("##") if token.startswith("##") else token
        else:
            if current_entity:
                entities.append((current_entity,current_type))
            current_entity = ""
            current_type = ""
    if current_entity:
        entities.append((current_entity,current_type))

    return entities
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    #1.加载数据,预处理
    train_dataset,val_dataset,bio_labels,id_to_label,label_to_id = get_dataset(tokenizer)
    #2.使用预训练的bert模型
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-chinese",num_labels = len(bio_labels),id2label = id_to_label,label2id = label_to_id
    )
    model.to(device)
    #3.构建数据加载器
    train_loader = DataLoader(train_dataset,batch_size = 16,shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size = 32)
    #4.训练
    train_evaluate(model,train_loader,val_loader,device)

    print("\n测试:")
    print(predict_ner("浙商银行企业信贷部叶老桂博士认为目前国内商业银行面临五大挑战",model,tokenizer,id_to_label,device))
    print(predict_ner("江苏警方通报特斯拉冲进店铺",model,tokenizer,id_to_label,device))
