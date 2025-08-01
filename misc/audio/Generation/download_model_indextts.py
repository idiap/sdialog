#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, UZH, JSALT (author: Srikanth Madikeri)
# Licensed under the Apache License, Version 2.0 (the "License");

from huggingface_hub import hf_hub_download

# model card is in https://huggingface.co/IndexTeam/IndexTTS-1.5/tree/main
file_names = [
    "README",
    "README.md",
    "bigvgan_discriminator.pth",  # https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/bigvgan_discriminator.pth
    "bigvgan_generator.pth",
    "bpe.model",
    "config.yaml",
    "dvae.pth",
    "gpt.pth",
    "unigram_12000.vocab",
]

for file_name in file_names:
    hf_hub_download(
        repo_id="IndexTeam/IndexTTS-1.5",
        filename=file_name,
        local_dir="./model/",
    )
