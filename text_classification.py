import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_scheduler
from datasets import ClassLabel, Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def parse_example(example):
    parts = example['text'].split("\t",1)
    return {'text':parts[1],'label':parts[0]}

def encode_label(example):
    return {"label":label_to_id[example["label"]]}

def preprocess(tokenizer):
    def process(example):
        result = tokenizer(example["text"],padding = "max_length",truncation = True,max_length = 128)
        return result
    return process

def train_evaluate(model,train_loader,val_loader,device):
    optimizer = torch.optim.Adam(model.parameters(),lr = 2e-5)
    epochs = 3
    train_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",optimizer = optimizer,
        num_warmup_steps = int(0.1 * train_steps),
        num_training_steps = train_steps
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader,desc=f"Epoch{epoch + 1}训练"):
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
        all_preds,all_labels = [],[]
        for batch in tqdm(val_loader,desc=f"Epoch{epoch + 1}验证"):
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            preds = torch.argmax(outputs.logits,dim = -1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch['labels'].cpu().tolist())

        acc = accuracy_score(all_labels,all_preds) * 100
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, 准确率 {acc:.2f}%")

#测试
def predict(text,model,tokenizer,id_to_label,device):
    inputs = tokenizer(text,return_tensors = "pt",truncation =True,max_length = 128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits,dim = -1).item()
    return id_to_label[pred]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #1.加载数据
    #print("加载数据集")
    dataset = Dataset.from_parquet("D:/learn/llm/nlp-basics/data/0000.parquet")

    #print(f"特征列: {dataset.features}")
    #print(f"样例: {dataset[0]}")
    #2.拆分标签和正文

    dataset = dataset.map(parse_example,remove_columns = ['text'])
    #print(f"样例:{dataset[0]}")

    #3.标签转数字
    labels = sorted(set(dataset['label']))
    #print(labels)
    label_to_id = { v:k for k,v in enumerate(labels)}
    #print(f"类别({len(labels)}):{labels}")

    dataset = dataset.map(encode_label)
    #print(dataset[0])

    #4.切分训练集验证集
    dataset = dataset.cast_column("label",ClassLabel(num_classes=len(labels)))
    dataset = dataset.train_test_split(test_size = 0.1,seed = 83,stratify_by_column = "label")
    print(f"训练集: {len(dataset['train'])}, 验证集: {len(dataset['test'])}")

    #5.加载模型和分词器
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels = len(labels))
    model.to(device)

    #6.数据预处理
    train_dataset = dataset["train"].map(preprocess(tokenizer),batched = True,load_from_cache_file = False)
    val_dataset = dataset["test"].map(preprocess(tokenizer),batched = True,load_from_cache_file = False)
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    train_dataset.set_format(type="torch",columns = ["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(train_dataset,batch_size = 16,shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size = 32)

    train_evaluate(model,train_loader,val_loader,device)
    id_to_label = {v:k for k,v in label_to_id.items()}
    print(f"「苹果发布新手机」→ {predict('苹果发布新手机', model, tokenizer, id_to_label, device)}")
    print(f"「国足今晚迎战日本」→ {predict('国足今晚迎战日本', model, tokenizer, id_to_label, device)}")
    print(f"「A股跌破3000点」→ {predict('A股跌破3000点', model, tokenizer, id_to_label, device)}")