1. 项目规划
项目名称：AI-Art-Generator-Pro
核心功能：用户输入提示词，调整 AI 参数，生成高质量图片。
前端界面：包含标题、学号（423830220）、姓名（邵嘉文）、输入框、参数滑块、生成按钮和结果展示区。
技术栈：Python + PyTorch + Diffusers + Gradio。
2. 环境准备
你需要安装以下库。建议在虚拟环境中安装：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # 如果有NVIDIA显卡
pip install diffusers transformers accelerate gradio safetensors
3. 项目代码
请在你的项目文件夹中创建一个名为 app.py 的文件，并将以下代码复制进去。
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr
import random

# --- 配置区域 ---
STUDENT_ID = "423830220"
STUDENT_NAME = "邵嘉文"
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # 使用经典的 v1.5 模型，兼容性好且效果好
# -----------------

class AIImageGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在初始化模型，使用设备: {self.device}...")
        
        # 加载模型
        # 使用 DPMSolverMultistepScheduler 调度器，这是一种更快的采样算法
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 切换调度器以获得更快的生成速度
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        
        # 如果是 GPU，开启内存优化
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            
        print("模型加载完成！")

    def generate(self, prompt, negative_prompt, steps, guidance_scale, seed, width, height):
        """
        核心生成函数
        """
        if self.device == "cuda":
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        else:
            generator = torch.Generator().manual_seed(int(seed))

        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
                width=int(width),
                height=int(height)
            )
            return result.images[0]
        except Exception as e:
            return None

# 初始化生成器实例
# 注意：在实际部署时，为了加快启动速度，可以只在第一次请求时加载模型
generator = AIImageGenerator()

def infer(prompt, negative_prompt, steps, guidance, seed, random_seed, width, height):
    """
    Gradio 接口函数
    """
    if random_seed:
        seed = random.randint(0, 2147483647)
    
    print(f"生成请求: 提示词='{prompt}', 种子={seed}")
    
    image = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance_scale=guidance,
        seed=seed,
        width=width,
        height=height
    )
    
    if image:
        return image, seed
    else:
        return None, seed

# --- 构建 Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # 1. 顶部 Header：显示学号和姓名
    gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>AI 智能绘图生成器</h1>
            <h3>学号: {STUDENT_ID} &nbsp;|&nbsp; 姓名: {STUDENT_NAME}</h3>
            <p>基于 Stable Diffusion v1.5 模型 | 输入描述，创造无限可能</p>
        </div>
    """)
    
    with gr.Row():
        # 左侧：输入控制区
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="正面提示词", placeholder="例如：一只穿着宇航服的猫，赛博朋克风格，超高清", lines=3)
            neg_prompt_input = gr.Textbox(label="负面提示词", placeholder="例如：模糊，低质量，变形，丑陋", lines=2)
            
            with gr.Accordion("高级参数设置", open=True):
                steps = gr.Slider(minimum=10, maximum=50, value=25, step=1, label="迭代步数")
                guidance = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="引导系数")
                width = gr.Slider(minimum=256, maximum=768, value=512, step=64, label="图片宽度")
                height = gr.Slider(minimum=256, maximum=768, value=512, step=64, label="图片高度")
                
                with gr.Row():
                    seed_input = gr.Number(label="随机种子", value=42)
                    random_seed = gr.Checkbox(label="随机种子", value=True)
            
            generate_btn = gr.Button("🚀 生成图片", variant="primary", size="lg")
            
            # 示例数据
            gr.Examples(
                examples=[
                    ["A futuristic city with flying cars, neon lights, cinematic lighting, 8k", "blurry, low quality", 25, 7.5, 512, 512, -1, True],
                    ["Portrait of a young woman, cyberpunk style, detailed face, neon background", "ugly, deformed, bad anatomy", 30, 8.0, 512, 768, 12345, False]
                ],
                inputs=[prompt_input, neg_prompt_input, steps, guidance, width, height, seed_input, random_seed]
            )

        # 右侧：输出展示区
        with gr.Column(scale=1):
            output_image = gr.Image(label="生成结果")
            output_seed = gr.Number(label="当前使用的种子")

    # 绑定事件
    generate_btn.click(
        fn=infer,
        inputs=[prompt_input, neg_prompt_input, steps, guidance, seed_input, random_seed, width, height],
        outputs=[output_image, output_seed]
    )

# 启动应用
if __name__ == "__main__":
    print(f"启动服务... 请在控制台查看本地访问地址 (通常是 http://127.0.0.1:7860)")
    demo.launch()
4. 如何运行与展示
本地运行：
在终端中进入项目目录，运行：python app.py
看到类似 Running on local URL: http://127.0.0.1:7860 的提示后，在浏览器打开该链接即可看到界面。
