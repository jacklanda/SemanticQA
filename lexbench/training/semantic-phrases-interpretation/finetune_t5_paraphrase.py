import os
import argparse
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, AutoModelForSeq2SeqLM

import eval_helper
import util
from util import NCParaphraseDataset


def main():
    parser = argparse.ArgumentParser(description="Noun Compound Interpretation Model")
    parser.add_argument(
        "--train", action="store_true", required=False, help="do train or not"
    )
    parser.add_argument(
        "--test", action="store_true", required=False, help="do train or not"
    )
    parser.add_argument(
        "--architechture",
        required=True,
        help="name of model architechture to use (huggingface model)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Specify the task to be performed: [iep, lcp, ncp]",
    )
    parser.add_argument("--lr", default=5e-5, help="learning rate")
    parser.add_argument("--bs", default=16, help="batch size")
    parser.add_argument("--epochs", default=4, help="number of epochs")
    parser.add_argument("--load", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", device)

    if args.task == "iep":
        train_df = util.load_train_df("data/idiom_train.csv")
        valid_df = util.load_test_valid_dataset("data/idiom_valid.csv")
        # test_df = util.load_test_df("data/idiom_test.csv")
        test_df = util.load_test_valid_dataset("data/idiom_test.csv")
    elif args.task == "lcp":
        train_df = util.load_train_df("data/collocation_train.csv")
        valid_df = util.load_test_valid_dataset("data/collocation_valid.csv")
        test_df = util.load_test_valid_dataset("data/collocation_test.csv")
    elif args.task == "ncp":
        train_df = util.load_train_df("data/train_gold.csv")
        valid_df = util.load_test_valid_dataset("data/valid_ds.csv")
        test_df = util.load_test_valid_dataset("data/test_gold.csv")

    train_dataset = NCParaphraseDataset(train_df["nc"], train_df["paraphrase"])
    train_loader = DataLoader(train_dataset, batch_size=int(args.bs))

    valid_dataset = NCParaphraseDataset(valid_df["nc"], valid_df["paraphrase"])
    valid_loader = DataLoader(valid_dataset, batch_size=int(args.bs))

    test_dataset = NCParaphraseDataset(test_df["nc"], test_df["paraphrase"])
    test_loader = DataLoader(
        test_dataset, batch_size=int(args.bs)
    )  # TODO: figure out how to change this from bs=1

    if "t5" in args.architechture:
        if args.load:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "./model_" + args.architechture, device_map="auto"
            )
            # model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.architechture)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.architechture, device_map="auto"
            )
            # model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.architechture)
    else:
        print("unsupported model")

    if args.train:
        train(model, tokenizer, train_loader, valid_loader, device, args)
        model.save_pretrained(os.getcwd() + "/model_" + args.architechture)

    if args.test:
        test(model, tokenizer, test_loader, device)


def train(model, tokenizer, train_loader, valid_loader, device, args):
    lr = float(args.lr)
    num_epochs = int(args.epochs)

    num_batches = len(train_loader)

    optim = Adafactor(model.parameters(), lr=lr, relative_step=False)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            optim.zero_grad()

            ncs, paraphrase = batch[0], batch[1]

            tokenized_ncs = tokenizer(
                ncs, padding=True, truncation=True, return_tensors="pt"
            )
            tokenized_paras = tokenizer(
                paraphrase, padding=True, truncation=True, return_tensors="pt"
            )

            input_ids = tokenized_ncs["input_ids"].to(device)
            attention_mask = tokenized_ncs["attention_mask"].to(device)

            labels = tokenized_paras["input_ids"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            epoch_loss += loss.item()

            loss.backward()
            optim.step()

        model.eval()
        valid_loss = 0
        valid_num_batches = len(valid_loader)
        with torch.no_grad():
            for batch in valid_loader:
                ncs, paraphras = batch[0], batch[1]

                tokenized_ncs = tokenizer(
                    ncs, padding=True, truncation=True, return_tensors="pt"
                )
                tokenized_paras = tokenizer(
                    paraphras, padding=True, truncation=True, return_tensors="pt"
                )

                input_ids = tokenized_ncs["input_ids"].to(device)
                attention_mask = tokenized_ncs["attention_mask"].to(device)
                labels = tokenized_paras["input_ids"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                valid_loss += loss.item()

        epoch_loss /= num_batches
        valid_loss /= valid_num_batches
        print(
            "epoch: "
            + str(epoch + 1)
            + ", train loss: "
            + str(epoch_loss)
            + ", valid loss: "
            + str(valid_loss)
        )


def test(model, tokenizer, test_loader, device):
    model.eval()
    with torch.no_grad():
        total_score = 0
        global_total_meteor = 0
        global_total_rougel = 0
        global_total_bertsc = 0
        global_total_semval = 0

        for idx, batch in enumerate(tqdm(test_loader), start=1):
            torch.cuda.empty_cache()

            # ncs, gold_paraphras = batch[0], batch[1][0].split("#*#*#*")  # gold_paraphras is a list: ["...", "...", "..."]
            ncs, gold_paraphras = (
                [nc for nc in batch[0]],
                [p for p in batch[1]],
            )  # gold_paraphras is a list: ["...", "...", "..."]
            print("ncs:", ncs)
            print("gold_paras:", gold_paraphras)

            # change from tuples to just strings
            # gold_paraphras = [p for p in gold_paraphras]

            tokenized_ncs = tokenizer(
                ncs, padding=True, truncation=True, return_tensors="pt"
            )

            input_ids = tokenized_ncs["input_ids"].to(device)

            num_paras = len(gold_paraphras)
            gen_paras = []

            for i in range(num_paras):
                output = generate_top_p(input_ids, model, p=0.9, t=0.7)
                gen_paras.append(tokenizer.decode(output, skip_special_tokens=True))

            print("gen_paras:")
            pprint(gen_paras)
            print("gold_paras:")
            pprint(gold_paraphras)
            # nc_average_score = eval_helper.batch_meteor(gen_paras, gold_paraphras)
            # total_score += nc_average_score

            assert len(gen_paras) == len(gold_paraphras)

            meteor_score = eval_helper.batch_meteor_score(gen_paras, gold_paraphras)
            rougel_score = eval_helper.batch_rougel_score(gen_paras, gold_paraphras)
            bertsc_score = eval_helper.batch_bertsc_score(gen_paras, gold_paraphras)
            semval_score = eval_helper.batch_semval_score(gen_paras, gold_paraphras)

            global_total_meteor += meteor_score
            global_total_rougel += rougel_score
            global_total_bertsc += bertsc_score
            global_total_semval += semval_score

            print("avg meteor:", global_total_meteor / idx)
            print("avg rougel:", global_total_rougel / idx)
            print("avg bertsc:", global_total_bertsc / idx)
            print("avg semval:", global_total_semval / idx)

        global_average_meteor = global_total_meteor / len(test_loader)
        global_average_rougel = global_total_rougel / len(test_loader)
        global_average_bertsc = global_total_bertsc / len(test_loader)
        global_average_semval = global_total_semval / len(test_loader)

        # average_score = total_score / len(test_loader)
        # print("test meteor score: ", average_score)
        print("avg meteor:", global_average_meteor)
        print("avg rougel:", global_average_rougel)
        print("avg bertsc:", global_average_bertsc)
        print("avg semval:", global_average_semval)


def generate_top_p(input_ids, model, p, t):
    tokenized_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=50,
        top_p=p,
        top_k=0,
        temperature=t,
    )
    return tokenized_output[0]


def generate_top_k(input_ids, model, k, temp=0):
    tokenized_output = model.generate(
        input_ids, do_sample=True, max_length=50, top_k=k, temperature=temp
    )
    return tokenized_output[0]


def generate_top_p_k(input_ids, model, p, k, temp=0):
    tokenized_output = model.generate(
        input_ids, do_sample=True, max_length=50, top_p=p, top_k=k, temperature=temp
    )
    return tokenized_output[0]


if __name__ == "__main__":
    main()
