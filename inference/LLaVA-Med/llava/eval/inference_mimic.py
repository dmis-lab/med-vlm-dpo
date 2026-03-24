import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import batch_tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


class LLaVAMedBatchInference:
    def __init__(self, args):
        self.args = args
        self.model_path = os.path.expanduser(args.model_path)
        self.model_name = get_model_name_from_path(self.model_path)

        set_seed(0)
        disable_torch_init()
        
        # 모델, 토크나이저, 이미지 프로세서 로드
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, args.model_base, self.model_name
        )

        # 토크나이저 패딩 설정
        self.tokenizer.padding_side = "left"
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)
        return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    
    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        return chunks[k]


    def process_batch(self, batch):
        """
        배치 내 각 데이터를 처리하여 모델 입력을 준비하는 함수
        """
        idx_list = []
        image_files = []
        prompts = []
        prompt_list = []
        answer_type = []

        for line in batch:
            idx_list.append(line["question_id"])
            # image_files.append(line["image_path"][0])
            image_files.append(line["image"])
            answer_type.append(line["answer_type"])
            # breakpoint()
            qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            # qs = "Illustrate the image through a descriptive explanation."
            
            if self.model.config.mm_use_im_start_end:
                qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
            else:
                qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"
            
            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            processed_prompt = conv.get_prompt()
            prompts.append(processed_prompt)
            prompt_list.append(qs)

        input_ids = batch_tokenizer_image_token(prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        # 토크나이징 및 패딩
        # input_ids = self.tokenizer(
        #     prompts,
        #     return_tensors='pt',
        #     padding="longest"
        # ).input_ids.cuda()

        # 이미지 로딩 및 전처리
        images = [Image.open(os.path.join(self.args.image_folder, img)) for img in image_files]
        image_tensors = process_images(images, self.image_processor, self.model.config).half().cuda()
        # breakpoint()

        attention_mask = (input_ids != 0).long().cuda()
        return idx_list, input_ids, attention_mask, image_tensors, prompt_list, answer_type
    
    def process_batch_long_form(self, batch):
        """
        배치 내 각 데이터를 처리하여 모델 입력을 준비하는 함수
        """
        idx_list = []
        image_files = []
        prompts = []
        gold_list = []

        for line in batch:
            idx_list.append(line["id"])
            # image_files.append(line["image_path"][0])
            image_files.append(line["image_path"][0])
            gold_list.append(line["report"])
            # breakpoint()
            
            qs = "Write a detailed report based on the image."
            
            if self.model.config.mm_use_im_start_end:
                qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
            else:
                qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"
            # breakpoint()
            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            processed_prompt = conv.get_prompt()
            prompts.append(processed_prompt)
            # breakpoint()


        input_ids = batch_tokenizer_image_token(prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        # 토크나이징 및 패딩
        # input_ids = self.tokenizer(
        #     prompts,
        #     return_tensors='pt',
        #     padding="longest"
        # ).input_ids.cuda()

        # 이미지 로딩 및 전처리
        images = [Image.open(os.path.join(self.args.image_folder, img)) for img in image_files]
        image_tensors = process_images(images, self.image_processor, self.model.config).half().cuda()
        # breakpoint()

        attention_mask = (input_ids != 0).long().cuda()
        # breakpoint()
        return idx_list, input_ids, attention_mask, image_tensors, gold_list
    
    def process_batch_long_amboss(self, batch):
        """
        배치 내 각 데이터를 처리하여 모델 입력을 준비하는 함수
        """
        idx_list = []
        image_files = []
        prompts = []
        gold_list = []

        for line in batch:
            # idx_list.append(line["id"])
            # image_files.append(line["image_path"][0])
            image_files.append(line["image_path"])
            gold_list.append(line["caption"])
            # breakpoint()
            
            qs = "Analyze the given medical image and provide a description, including relevant clinical findings, abnormalities, and potential diagnoses."
            
            if self.model.config.mm_use_im_start_end:
                qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
            else:
                qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"
            # breakpoint()
            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            processed_prompt = conv.get_prompt()
            prompts.append(processed_prompt)
            # breakpoint()


        input_ids = batch_tokenizer_image_token(prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        # 토크나이징 및 패딩
        # input_ids = self.tokenizer(
        #     prompts,
        #     return_tensors='pt',
        #     padding="longest"
        # ).input_ids.cuda()

        # 이미지 로딩 및 전처리
        images = [Image.open(os.path.join(self.args.image_folder, img)) for img in image_files]
        image_tensors = process_images(images, self.image_processor, self.model.config).half().cuda()
        # breakpoint()

        attention_mask = (input_ids != 0).long().cuda()
        # breakpoint()
        return idx_list, input_ids, attention_mask, image_tensors, gold_list


    def run_inference(self, input_ids, attention_mask, image_tensors):
        """
        모델 추론을 실행하는 함수
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensors,
                do_sample=self.args.temperature > 0,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=1024,
                use_cache=True
            )
        
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # breakpoint()
        return outputs
    
    def clip_process_batch_top_k(self, data_batch, top_num):
        """
        주어진 데이터 배치에서 reference 이미지 및 top-k 후보 이미지를 로드하고 모델 입력 데이터로 변환.
        """

        idx_list = []
        candidate_images = []
        prompts = []
        prompt_list = []    

        for data in data_batch:

            idx_list.append(data["reference_image"]["id"])

            can_image_path = data["selected_candidates"][f"top_{top_num}"]["candidate_image"]

            candidate_images.append(can_image_path)

            qs = data["reference_image"]["conversation"]["human"]

            if self.model.config.mm_use_im_start_end:
                qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
            else:
                qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            processed_prompt = conv.get_prompt()
            prompts.append(processed_prompt)
            prompt_list.append(qs)

        input_ids = batch_tokenizer_image_token(prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        # 토크나이징 및 패딩
        # input_ids = self.tokenizer(
        #     prompts,
        #     return_tensors='pt',
        #     padding="longest"
        # ).input_ids.cuda()

        # 이미지 로딩 및 전처리
        images = [Image.open(os.path.join(self.args.image_folder, img)) for img in candidate_images]
        image_tensors = process_images(images, self.image_processor, self.model.config).half().cuda()

        attention_mask = (input_ids != 0).long().cuda()

        return idx_list, input_ids, attention_mask, image_tensors, prompt_list, data_batch
    
    def clip_process_batch_long_form_top_k(self, data_batch, top_num):
        """
        주어진 데이터 배치에서 reference 이미지 및 top-k 후보 이미지를 로드하고 모델 입력 데이터로 변환.
        """

        idx_list = []
        candidate_images = []
        prompts = []
        prompt_list = []    

        for data in data_batch:

            idx_list.append(data["reference_image"]["image"])

            can_image_path = data["selected_candidates"][f"top_{top_num}"]["candidate_image"]

            candidate_images.append(can_image_path)

            ######### image captioning ########
            qs = "Illustrate the image through a descriptive explanation."
            #########  radio report ###########
            # qs = "Write a detailed report based on the image."

            if self.model.config.mm_use_im_start_end:
                qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
            else:
                qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            processed_prompt = conv.get_prompt()
            prompts.append(processed_prompt)
            prompt_list.append(qs)

        input_ids = batch_tokenizer_image_token(prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        # 토크나이징 및 패딩
        # input_ids = self.tokenizer(
        #     prompts,
        #     return_tensors='pt',
        #     padding="longest"
        # ).input_ids.cuda()

        # 이미지 로딩 및 전처리
        images = [Image.open(os.path.join(self.args.image_folder, img)) for img in candidate_images]
        image_tensors = process_images(images, self.image_processor, self.model.config).half().cuda()

        attention_mask = (input_ids != 0).long().cuda()
        # breakpoint()

        return idx_list, input_ids, attention_mask, image_tensors, prompt_list, data_batch
    
    def clip_process_batch_long_form_top_k_weak_gt(self, data_batch, top_num):
        """
        주어진 데이터 배치에서 reference 이미지 및 top-k 후보 이미지를 로드하고 모델 입력 데이터로 변환.
        """

        idx_list = []
        candidate_images = []
        prompts = []
        prompt_list = []    

        for data in data_batch:

            idx_list.append(data["reference_image"]["image"])

            can_image_path = data["selected_candidates"][f"top_{top_num}"]["candidate_image"]

            candidate_images.append(can_image_path)

            reference_text = next(conv['value'] for conv in data['selected_candidates'][f'top_{top_num}']['caption'] if conv['from'] == 'gpt')
            ######### image captioning ########
            instruction = "Illustrate the image through a descriptive explanation."
            #########  radio report ###########
            # instruction = "Write a detailed report based on the image."

            qs = f"""Reference Text: {reference_text}\n {instruction}"""

            if self.model.config.mm_use_im_start_end:
                qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
            else:
                qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            processed_prompt = conv.get_prompt()
            prompts.append(processed_prompt)
            prompt_list.append(qs)

        input_ids = batch_tokenizer_image_token(prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        # 토크나이징 및 패딩
        # input_ids = self.tokenizer(
        #     prompts,
        #     return_tensors='pt',
        #     padding="longest"
        # ).input_ids.cuda()

        # 이미지 로딩 및 전처리
        images = [Image.open(os.path.join(self.args.image_folder, img)) for img in candidate_images]
        image_tensors = process_images(images, self.image_processor, self.model.config).half().cuda()

        attention_mask = (input_ids != 0).long().cuda()
        # breakpoint()

        return idx_list, input_ids, attention_mask, image_tensors, prompt_list, data_batch
    
    def clip_process_batch_top_k_weak_gt(self, data_batch, top_num):
        """
        주어진 데이터 배치에서 reference 이미지 및 top-k 후보 이미지를 로드하고 모델 입력 데이터로 변환.
        """

        idx_list = []
        candidate_images = []
        prompts = []
        prompt_list = []    

        for data in data_batch:

            idx_list.append(data["reference_image"]["id"])

            can_image_path = data["selected_candidates"][f"top_{top_num}"]["candidate_image"]

            candidate_images.append(can_image_path)

            reference_text = next(conv['value'] for conv in data['selected_candidates'][f'top_{top_num}'][f'top_{top_num}_alignment_conversations'] if conv['from'] == 'gpt')

            instruction = data["reference_image"]["conversation"]["human"]

            qs = f"""Reference Text: {reference_text}
            {instruction}
            """

            if self.model.config.mm_use_im_start_end:
                qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
            else:
                qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            processed_prompt = conv.get_prompt()
            prompts.append(processed_prompt)
            prompt_list.append(qs)

        # input_ids = batch_tokenizer_image_token(prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        input_ids = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding="longest"
        ).input_ids.cuda()

        images = [Image.open(os.path.join(self.args.image_folder, img)) for img in candidate_images]
        image_tensors = process_images(images, self.image_processor, self.model.config).half().cuda()
        # breakpoint()
        ###################################
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        ###################################

        attention_mask = (input_ids != 0).long().cuda()
        # breakpoint()

        return idx_list, input_ids, attention_mask, image_tensors, prompt_list, data_batch
    

    def batch_eval(self):
        """
        전체 배치를 평가하고 JSON 파일로 결과를 저장하는 함수
        """
        
        questions = [json.loads(q) for q in open(os.path.expanduser(self.args.question_file), "r")]
        questions = self.get_chunk(questions, self.args.num_chunks, self.args.chunk_idx)

        batch_size = self.args.batch_size
        answers_file = os.path.expanduser(self.args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        results = []

        for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):
            batch = questions[i:i + batch_size]

            # 모델 입력 데이터 생성
            idx_list, input_ids, attention_mask, image_tensors, prompt_list, answer_type = self.process_batch(batch)

            # 모델 추론 수행
            outputs = self.run_inference(input_ids, attention_mask, image_tensors)
            # print(self.tokenizer.padding_side)
            # print(self.tokenizer.pad_token)
            # breakpoint()
            # 결과 저장
            for idx, output, original_prompt, answer_type in zip(idx_list, outputs, prompt_list, answer_type):
                ans_id = shortuuid.uuid()
                results.append({
                    "question_id": idx,
                    "prompt": original_prompt,
                    "text": output.strip(),
                    "answer_id": ans_id,
                    "model_id": self.model_name,
                    "answer_type": answer_type,
                    "metadata": {}
                })
                print(output)
                
        # JSON 파일 저장
        with open(answers_file, "w") as ans_file:
            for res in results:
                ans_file.write(json.dumps(res) + "\n")

    def batch_eval_long_form(self):
        """
        전체 배치를 평가하고 JSON 파일로 결과를 저장하는 함수
        """
        
        questions = [json.loads(q) for q in open(os.path.expanduser(self.args.question_file), "r")]
        questions = self.get_chunk(questions, self.args.num_chunks, self.args.chunk_idx)

        batch_size = self.args.batch_size
        answers_file = os.path.expanduser(self.args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        results = []

        for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):
            batch = questions[i:i + batch_size]

            # 모델 입력 데이터 생성
            idx_list, input_ids, attention_mask, image_tensors, gold_list = self.process_batch_long_form(batch)

            # 모델 추론 수행
            outputs = self.run_inference(input_ids, attention_mask, image_tensors)
            # print(self.tokenizer.padding_side)
            # print(self.tokenizer.pad_token)
            # breakpoint()
            # 결과 저장
            for idx, output, gold_prompt in zip(idx_list, outputs, gold_list):
                ans_id = shortuuid.uuid()
                results.append({
                    "id": idx,
                    "gold_caption": gold_prompt,
                    "caption_generated": output.strip(),
                    "answer_id": ans_id,
                    "model_id": self.model_name,
                    "metadata": {}
                })
                print(output)
                
        # JSON 파일 저장
        with open(answers_file, "w") as ans_file:
            for res in results:
                ans_file.write(json.dumps(res) + "\n")

    def batch_eval_long_amboss(self):
        """
        전체 배치를 평가하고 JSON 파일로 결과를 저장하는 함수
        """
        
        questions = [json.loads(q) for q in open(os.path.expanduser(self.args.question_file), "r")]
        questions = self.get_chunk(questions, self.args.num_chunks, self.args.chunk_idx)

        batch_size = self.args.batch_size
        answers_file = os.path.expanduser(self.args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        results = []

        for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):
            batch = questions[i:i + batch_size]

            # 모델 입력 데이터 생성
            idx_list, input_ids, attention_mask, image_tensors, gold_list = self.process_batch_long_amboss(batch)

            # 모델 추론 수행
            outputs = self.run_inference(input_ids, attention_mask, image_tensors)
            # print(self.tokenizer.padding_side)
            # print(self.tokenizer.pad_token)
            # breakpoint()
            # 결과 저장
            for output, gold_prompt in zip(outputs, gold_list):
                ans_id = shortuuid.uuid()
                results.append({
                    "gold_caption": gold_prompt,
                    "caption_generated": output.strip(),
                    "answer_id": ans_id,
                    "model_id": self.model_name,
                    "metadata": {}
                })
                print(output)
                
        # JSON 파일 저장
        with open(answers_file, "w") as ans_file:
            for res in results:
                ans_file.write(json.dumps(res) + "\n")


    def clip_batch_long_form_inference_top_k(self):
        # (self, json_file_path, batch_size, base_dir, output_file, top_num)
        """
        CLIP 기반 top-k 후보 이미지 비교를 위한 배치 추론 함수
        """

        top_num = self.args.top_num        

        questions = [json.loads(q) for q in open(os.path.expanduser(self.args.question_file), "r")]
        questions = self.get_chunk(questions, self.args.num_chunks, self.args.chunk_idx)

        batch_size = self.args.batch_size
        answers_file = os.path.expanduser(self.args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        with open(answers_file, "w") as outfile:
            for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):

                batch = questions[i:i + batch_size]

                idx_list, input_ids, attention_mask, image_tensors, prompt_list, input = self.clip_process_batch_long_form_top_k(batch, top_num)

                outputs = self.run_inference(input_ids, attention_mask, image_tensors)

                for i, data in enumerate(input):
                    output_text = outputs[i].strip()
                    output_data_load = {
                        "reference_image" : data["reference_image"]["image"],
                        f"top_{top_num}_image" : data["selected_candidates"][f"top_{top_num}"]["candidate_image"],
                        f"top_{top_num}_caption" : data["selected_candidates"][f"top_{top_num}"]["caption"],
                        "similarity" : data["selected_candidates"][f"top_{top_num}"]["similarity"],
                        f"top_{top_num}_output" : output_text
                    }
                    # breakpoint()
                    print(output_text)
                    outfile.write(json.dumps(output_data_load) + "\n")
                    outfile.flush()
                

        print(f"✅ CLIP Top-{top_num} Inference 완료!")

    def clip_batch_long_form_inference_top_k_weak_gt(self):
        # (self, json_file_path, batch_size, base_dir, output_file, top_num)
        """
        CLIP 기반 top-k 후보 이미지 비교를 위한 배치 추론 함수
        """

        top_num = self.args.top_num        

        questions = [json.loads(q) for q in open(os.path.expanduser(self.args.question_file), "r")]
        questions = self.get_chunk(questions, self.args.num_chunks, self.args.chunk_idx)

        batch_size = self.args.batch_size
        answers_file = os.path.expanduser(self.args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        ########
        def get_text_length(data):
            reference_text = next(conv['value'] for conv in data['selected_candidates'][f'top_{top_num}']['caption'] if conv['from'] == 'gpt')
            # instruction = "Illustrate the image through a descriptive explanation."
            return len(reference_text)

        questions.sort(key=get_text_length)  # 텍스트 길이 기준 정렬

        with open(answers_file, "w") as outfile:
            for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):

                batch = questions[i:i + batch_size]

                idx_list, input_ids, attention_mask, image_tensors, prompt_list, input = self.clip_process_batch_long_form_top_k_weak_gt(batch, top_num)

                outputs = self.run_inference(input_ids, attention_mask, image_tensors)

                for i, data in enumerate(input):
                    output_text = outputs[i].strip()
                    output_data_load = {
                        "reference_image" : data["reference_image"]["image"],
                        f"top_{top_num}_image" : data["selected_candidates"][f"top_{top_num}"]["candidate_image"],
                        f"top_{top_num}_caption" : data["selected_candidates"][f"top_{top_num}"]["caption"],
                        "similarity" : data["selected_candidates"][f"top_{top_num}"]["similarity"],
                        f"top_{top_num}_output_weak_gt" : output_text
                    }
                    # breakpoint()
                    print(output_text)
                    outfile.write(json.dumps(output_data_load) + "\n")
                    outfile.flush()
                

        print(f"✅ CLIP Top-{top_num} Inference 완료!")


    def clip_batch_inference_top_k(self):
        # (self, json_file_path, batch_size, base_dir, output_file, top_num)
        """
        CLIP 기반 top-k 후보 이미지 비교를 위한 배치 추론 함수
        """

        top_num = self.args.top_num        

        questions = [json.loads(q) for q in open(os.path.expanduser(self.args.question_file), "r")]
        questions = self.get_chunk(questions, self.args.num_chunks, self.args.chunk_idx)

        batch_size = self.args.batch_size
        answers_file = os.path.expanduser(self.args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        def get_text_length(data):
            instruction = data["reference_image"]["conversation"]["human"]
            return len(instruction)

        questions.sort(key=get_text_length)  # 텍스트 길이 기준 정렬

        with open(answers_file, "w") as outfile:
            for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):

                batch = questions[i:i + batch_size]

                idx_list, input_ids, attention_mask, image_tensors, prompt_list, input = self.clip_process_batch_top_k(batch, top_num)

                outputs = self.run_inference(input_ids, attention_mask, image_tensors)

                for i, data in enumerate(input):
                    output_text = outputs[i].strip()
                    output_data_load = {
                        "reference_image" : data["reference_image"]["image"],
                        f"top_{top_num}_image" : data["selected_candidates"][f"top_{top_num}"]["candidate_image"],
                        "ref_instructiontuning_conversations" : data["reference_image"]["conversation"],
                        f"top_{top_num}_caption" : data["selected_candidates"][f"top_{top_num}"][f"top_{top_num}_alignment_conversations"],
                        "similarity" : data["selected_candidates"][f"top_{top_num}"]["similarity"],
                        f"top_{top_num}_output" : output_text
                    }
                    # breakpoint()
                    print(output_text)
                    outfile.write(json.dumps(output_data_load) + "\n")
                    outfile.flush()
                

        print(f"✅ CLIP Top-{top_num} Inference 완료!")


    def clip_batch_inference_top_k_weak_gt(self):
        # (self, json_file_path, batch_size, base_dir, output_file, top_num)
        """
        CLIP 기반 top-k 후보 이미지 비교를 위한 배치 추론 함수
        """

        top_num = self.args.top_num

        questions = [json.loads(q) for q in open(os.path.expanduser(self.args.question_file), "r")]
        questions = self.get_chunk(questions, self.args.num_chunks, self.args.chunk_idx)

        batch_size = self.args.batch_size
        answers_file = os.path.expanduser(self.args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        def get_text_length(data):
            reference_text = next(conv['value'] for conv in data['selected_candidates'][f'top_{top_num}'][f'top_{top_num}_alignment_conversations'] if conv['from'] == 'gpt')
            instruction = data["reference_image"]["conversation"]["human"]
            return len(reference_text) + len(instruction)

        questions.sort(key=get_text_length)  # 텍스트 길이 기준 정렬

        with open(answers_file, "w") as outfile:
            for i in tqdm(range(0, len(questions), batch_size), desc="Processing Batches"):

                batch = questions[i:i + batch_size]

                idx_list, input_ids, attention_mask, image_tensors, prompt_list, input = self.clip_process_batch_top_k_weak_gt(batch, top_num)

                outputs = self.run_inference(input_ids, attention_mask, image_tensors)
                # breakpoint()
                for i, data in enumerate(input):
                    output_text = outputs[i].strip()
                    output_data_load = {
                        "reference_image" : data["reference_image"]["image"],
                        f"top_{top_num}_image" : data["selected_candidates"][f"top_{top_num}"]["candidate_image"],
                        "ref_instructiontuning_conversations" : data["reference_image"]["conversation"],
                        f"top_{top_num}_caption" : data["selected_candidates"][f"top_{top_num}"][f"top_{top_num}_alignment_conversations"],
                        "similarity" : data["selected_candidates"][f"top_{top_num}"]["similarity"],
                        f"top_{top_num}_output_weakgt" : output_text
                    }
                    
                    print(output_text)
                    outfile.write(json.dumps(output_data_load) + "\n")
                    outfile.flush()
                

        print(f"✅ CLIP Top-{top_num} Inference 완료!")

    def clip_batch_inference_top_k_weak_gt_filtered(self):
        # (self, json_file_path, batch_size, base_dir, output_file, top_num)
        """
        CLIP 기반 top-k 후보 이미지 비교를 위한 배치 추론 함수
        """

        top_num = self.args.top_num

        questions = [json.loads(q) for q in open(os.path.expanduser(self.args.question_file), "r")]
        questions = self.get_chunk(questions, self.args.num_chunks, self.args.chunk_idx)

        batch_size = self.args.batch_size
        answers_file = os.path.expanduser(self.args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        def get_text_length(data):
            reference_text = next(conv['value'] for conv in data['selected_candidates'][f'top_{top_num}'][f'top_{top_num}_alignment_conversations'] if conv['from'] == 'gpt')
            instruction = data["reference_image"]["conversation"]["human"]
            return len(reference_text) + len(instruction)

        questions.sort(key=get_text_length) 

        with open(answers_file, "w") as outfile:
            data_batch = []
            total_lines = len(questions)

            for item in tqdm(questions, total = total_lines, desc="Processing Batches", unit="sample"):

                if item["selected_candidates"][f"top_{top_num}"][f"top_{top_num}_rejected_weak_gt"] != "":
                    continue

                data_batch.append(item)
                # breakpoint()

                if len(data_batch) == batch_size:
                    idx_list, input_ids, attention_mask, image_tensors, prompt_list, input = self.clip_process_batch_top_k_weak_gt(data_batch, top_num)

                    outputs = self.run_inference(input_ids, attention_mask, image_tensors)
                    # breakpoint()
                    for i, data in enumerate(input):
                        output_text = outputs[i].strip()
                        output_data_load = {
                            "reference_image" : data["reference_image"]["image"],
                            f"top_{top_num}_image" : data["selected_candidates"][f"top_{top_num}"]["candidate_image"],
                            "ref_instructiontuning_conversations" : data["reference_image"]["conversation"],
                            f"top_{top_num}_caption" : data["selected_candidates"][f"top_{top_num}"][f"top_{top_num}_alignment_conversations"],
                            "similarity" : data["selected_candidates"][f"top_{top_num}"]["similarity"],
                            f"top_{top_num}_output_weakgt" : output_text
                        }
                    
                        print(output_text)
                        outfile.write(json.dumps(output_data_load) + "\n")
                        outfile.flush()
                    data_batch = []

            if data_batch:
                idx_list, input_ids, attention_mask, image_tensors, prompt_list, input = self.clip_process_batch_top_k_weak_gt(data_batch, top_num)
                outputs = self.run_inference(input_ids, attention_mask, image_tensors)
                    # breakpoint()
                for i, data in enumerate(input):
                    output_text = outputs[i].strip()
                    output_data_load = {
                        "reference_image" : data["reference_image"]["image"],
                        f"top_{top_num}_image" : data["selected_candidates"][f"top_{top_num}"]["candidate_image"],
                        "ref_instructiontuning_conversations" : data["reference_image"]["conversation"],
                        f"top_{top_num}_caption" : data["selected_candidates"][f"top_{top_num}"][f"top_{top_num}_alignment_conversations"],
                        "similarity" : data["selected_candidates"][f"top_{top_num}"]["similarity"],
                        f"top_{top_num}_output_weakgt" : output_text
                    }
                    
                    print(output_text)
                    outfile.write(json.dumps(output_data_load) + "\n")
                    outfile.flush()

        print(f"✅ CLIP Top-{top_num} Inference 완료!")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-num", type=int, default=None)

    args = parser.parse_args()

    evaluator = LLaVAMedBatchInference(args)

    evaluator.batch_eval_long_form()
    
