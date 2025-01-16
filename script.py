import random
import os
import logging
from typing import List, Dict
import numpy as np
import torch
import vllm
from vllm import SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Set environment variables for CuBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"

# Model configuration
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# Load the model (vllm.LLM does not have `parameters`, so no need to modify it)
model = vllm.LLM(
    model=MODEL_NAME,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=6144,
    enforce_eager=True,
)

# Function to create prompts
def make_prompt(prompt: str) -> str:
    system_msg = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{{{ You are a helpful AI assistant }}}}<|eot_id|>"
    user_msg = f"<|start_header_id|>user<|end_header_id|>\n{{{{ {prompt} }}}}<|eot_id|>"
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
    return f"{system_msg}{user_msg}{assistant_start}"

# Function to generate responses
def generate_responses(
    model: vllm.LLM, prompts: List[str], sampling_params: SamplingParams
) -> Dict[str, str]:
    requests = [make_prompt(prompt) for prompt in prompts]
    responses = model.generate(requests, sampling_params, use_tqdm=True)
    return {
        prompt: response.outputs[0].text for prompt, response in zip(prompts, responses)
    }

# Set deterministic behavior
def set_deterministic(seed: int, *, any_card: bool = False):
    log.info(f"Setting deterministic mode with seed {seed!r}, any_card={any_card!r}")
    random.seed(seed)
    np.random.seed(seed)
    set_deterministic_pytorch(seed)

def set_deterministic_pytorch(seed: int):
    # Set CUBLAS workspace config
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace_config is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set PyTorch deterministic settings
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True) # alternative : torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.utils.deterministic.fill_uninitialized_memory = True

    # Disable TensorFloat32 for consistent precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU

# Solve prompt and generate deterministic responses
def solve_prompt(payload):
    # Make run deterministic
    set_deterministic(int(payload["seed"]))
    sampling_params = SamplingParams(
        max_tokens=int(payload["max_tokens"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        seed=int(payload["seed"]),
    )
    prompts = payload["prompts"].splitlines()

    log.info(f"Generating {len(prompts)} responses with seed {payload['seed']}")
    responses = generate_responses(model, prompts, sampling_params)

    return responses

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

# Example payload (parsed from JSON)
payload = {
    "prompts": "Who won the most Oscars among the actors listed?\nQ: What is a common method used in finance to estimate the impact of a specific change in the price of a commodity on a company's profitability?\nWhat historical cultures are known for their basket weaving techniques?\nWhat is the impact of natural light in landscape photography?\nWhat role does personal experience play in a person's adoption of atheism?\nWhat impact do IoT technologies have on traffic management in smart cities?\nWho is considered the father of the gods in Greek mythology?\nWhat is the role of typography in graphic design?\nWhat is the role of coral in an aquarium ecosystem?\nWhat is the traditional process for verifying direct lineage in genealogical research?\nWhat is the purpose of individuals participating in flash mobs often wearing strange outfits and performing unusual actions?\nWhat is the spin of an electron and how does it affect its quantum behavior?\nWhat factors might influence the ability to predict the emergence of a viral pandemic?\nWhat was the role of television in propagating the \"cold war\" ideology during the 1950s?\nWhat was one significant impact of the Renaissance on daily life in Italy?\nWhat is the mechanism of action of opioids in pain relief?\nWhat is the primary objective of fortification in military science?\nWhat is the expected value of a random variable within a Poisson process if the average rate of occurrence is known?\nWhat different perspectives exist within Christian salvation?\nWhat are some potential effects of nootropics on cognitive functions?\nWhy is string theory significant in the effort to unify quantum mechanics and general relativity?\nWhat methods are commonly used to debug a complex software application?\nCan you explain how living tissue is engineered for tissue repair or replacement?\nWhat are some common materials used in spinning to create traditional textiles like tweed or tweed-like fabrics?\nWhat is a real-world application of collective intelligence in problem-solving?\nWhat are the main factors that affect the growth rate of a sunflower?\nWhat are the key indicators used to evaluate a company's financial health?\nWhat challenges might a social worker typically encounter when counseling individuals from diverse backgrounds?\nWhat role does diversification play in risk management within an investment portfolio?\nWhy does a kite fly at a particular height before it stabilizes?\nWhy is neopaganism often associated with environmentalism and nature spirituality?\nWhat influences have shaped the narrative style commonly found in underground comics during the 1960s and 1970s?\nWhat is the role of stratigraphy in archaeological site excavation?\nWhat is the role of bees in pollination?\nWhat were the first three challenges you faced upon starting your travel blog and how did you address them?\nWhich property of an active pharmaceutical ingredient is crucial for its absorption into the bloodstream?\nCan you explain what signifiers are to language signs?\nWhat are common separation techniques used after distillation?\nWhat plants are traditionally used in Chaga mushroom medicine, and how are they prepared for use?\nWhat is the role of warp drives in faster-than-light travel in a hypothetical hard science fiction novel?\nWhat are the primary causes of desertification that contribute to land degradation?\nWhat happens when lava cools down rapidly upon reaching the Earth's surface?\nCan you describe the concept of dualism in Zoroastrianism?\nWhat are the key components of interpretive dance?\nWhat exposure pathway is typically NOT associated with acute toxicity symptoms in humans?\nWhat charge does a proton have?\nWhat impact has digital technology had on humanities scholarship according to your perspective?\nWhat effect does smoking have on lung health in the short term?\nWhat role does molecular gastronomy play in modernist cuisine, and how has it revolutionized the textures of certain food items?\nWhat materials are typically used to improve acoustics in a recording studio?\nWhat role do bees play in the ecosystem?\nWhat were the primary causes of World War I, and how did these factors interplay?\nWhat holidays are predominantly celebrated in the United States?\nWhat is the cochlea's role in our perception of pitch?\nWhat is the main means of transmitting data in the internet?\nWhat role do spices play in traditional Indian cuisine?\nWhat is Pantheism and how does it differ from traditional theism?\nWhat are some ways consumers can contribute to circular economy initiatives?\nWhat are the main challenges to human rights in democratic societies?\nWhat are the main themes addressed by Virginia Woolf in her novel \"Mrs. Dalloway\"?\nWhat are some roles of emotional support in palliative care for patients and their families?\nWhat are the common barriers to accessing healthcare in urban areas?\nQ: What is the role of Euclidean geometry in understanding the properties of basic shapes like triangles and circles?\nWhat is the central concept behind ecological art and how does it aim to impact viewer awareness and behavior towards the environment?\nWhat is the difference between arterial blood gases (ABG) and complete blood count (CBC)?\nWhat kind of landscapes are often represented on a cartographic map of a mountainous region?\nQuestion 1: What are the two major types of photosynthesis and how are they different from each other?\nWhy do metals tend to be good conductors of electricity?\nWhat are some materials commonly used in body painting and temporary art forms?\nWho is the main antagonist in the first part of \"Sword Art Online\"?\nWhat role did Bushido play in the daily life of a samurai, aside from combat?\nWhat are the key elements that define the aesthetic experience in art appreciation?\nWhat is a real-world scenario where solving a system of linear equations might be necessary?\nWhat is a tool commonly used in a kitchen that not only helps in chopping vegetables but also in blending soup?\nWhat is the deepest known point one can dive safely without decompression stops?\nWhat is the believed effect of quartz on emotional stability?\nHow might psychological stress influence physiological pain perception?\nWhat are the primary modes of transportation used in New York City?\nWhat role do non-state actors play in the concept of collective security within international relations?\nCan you explain why it's important to act ethically in business dealings?\nWhat is the main purpose of integrating meditative elements into the physical layout of a labyrinth's walking paths?\nWho are the primary belligerents in the World War I and II, and what caused the outbreak of these wars?\nWhat strategies are politicians known to use to connect with their constituents and garner support?\nWhat is your morning routine you follow to ensure your well-being?\nWhat is the famous wordplay on the phrase \"metro\" that relates to hidden city exploration?\nWhat is a basic application of nanocomputers in medicine that has been researched or demonstrated?\nQuestion: What are some primary sources of news in today's digital age?\nCan you describe how a 'ping sweep' can influence game-day strategies in football?\nWhat is one real-world scenario where a company successfully implemented a SWOT analysis to improve a strategic decision?\nWhat role do hormones play in butterfly metamorphosis?\nWhat are the challenges faced by physicists in confirming the existence of new particles beyond W and Z bosons, and how are experimental results currently addressing these challenges?\nWhich renewable energy source do you think should be prioritized in urban areas: solar, wind, or hydropower?\nWhat are the main steps involved in carpentry projects?\nWhat role does the Sonic hedgehog (Shh) protein play in limb development?\nWhat are cranes designed for in construction sites?\nWhat are some common examples of symptoms that might occur after intentional poisoning?\nWhat statistical measure is commonly used in psychometrics to assess the reliability of a psychological test?\nWhat role does ventilation play in the mining process?\nWhat role does the concept of freedom play in independent radio and underground broadcasts?\nWhat is the role of harmony in musical performances?\nWhat is the primary role of a chemical engineer in a petrochemical refinery?\nWhat are some key factors that motivated individuals to engage in independent zine publication?\nWhat common challenges might orchid cultivators face when growing rare orchids?\nWhat are the key elements to consider in post-production editing for a short film?\nWhat are some common themes explored in slow cinema that contribute to its unique storytelling approach?\nWhat are the common themes in urban exploration literature based on abandoned places?\nWhat streaming platform was predominantly used before the advent of YouTube?\nWhat is the mean of a dataset?\nWhat impact does the intention behind the photograph composition have on the final aesthetic perception of modern ruins?\nWhat is a common festival celebrated in many Pagan traditions involving a bonfire and dance?\nWhat is the impact of underground comics on mainstream pop culture?\nWhat is the main theme represented in the movie \"The Shawshank Redemption,\" and how is it conveyed through the experiences of the protagonist?\nWhat is the main practice that distinguishes Tenrikyo's spiritual path from other religions?\nWhat is the purpose of a resistor in a circuit?\nWhy is the mean considered a valuable measure in statistics when analyzing data sets?\nCan you explain the basic principle of fiber-to-the-home (FTTH)?\nWhat is the average lifespan of a red blood cell in humans?\nWhat is the historical significance of ikat patterns in traditional Southeast Asian textiles?\nWhich historical figure is often credited with popularizing the concept of a full-scale carnival?\nWhat is the main concept behind the Pythagorean theorem?\nWhat is the most common initial clinical sign of feline diabetes mellitus?\nWhy is regular physical activity important for mental health?\nWhat is the role of Lakshmi in Hindu mythology and how is she usually portrayed?\nWhat famous historical event heavily relied on cryptology for secure communication?\nWhat common health benefits are often associated with chromium's role in nutraceuticals?\nWhat is the role of simulation modeling in Industrial Engineering?\nWhat is the role of fungi in an ecosystem?\nWhat role does Zakat play in social welfare according to Islamic teachings?\nWhat is one advantage of alternative schooling models over traditional schooling for student engagement?\nWhat role does accountability play in public administration?\nWho can be considered a key beneficiary from successful data storytelling in a company context?\nWhat was a key characteristic of Renaissance art?\nWho is known to have developed the first fully implantable device and what was the device?\nWhat are the key trends influencing consumer behavior in social media marketing in 2023?\nWhat are mRNA vaccines and how do they work in the body to protect against diseases?\nWhich text by Joseph Campbell's monomyth can be linked to esotericism and is often reflected in various cultural narratives worldwide?\nWho produces antibodies specifically targeting antigens our body encounters?\nWhat is an example of a successful city hack that utilized tactical urbanism principles to improve a public space?\nWhy is it important for researchers to conduct empirical studies?\nQ: How does cognitive computing enhance user experience?\nWhat is a real-world application of brain-computer interfaces outside of medical rehabilitation?\nWhat are the known factors that contribute to viral evolution and how can they accelerate a pandemic?\nWhat is the fundamental theorem of calculus?\nWhat instruments can be commonly found in a jazz ensemble?\nWhat does crossing your arms usually mean in a non-verbal conversation?\nWhat is the basic shape of a capsid and how does it protect viral genetic material?\nWhat is the importance of shrimp in marine ecosystems?\nWhat role does neuroplasticity play in memory formation?\nWhat are the primary types of catheters used during an interventional cardiology procedure?\nWhat causes the aurora phenomena on Jupiter's poles?\nWhat is the property of a rectangle opposite each other equal to its length?\nWhat themes are you exploring in your writing project?\nWhat is the difference between analog and digital video recording formats?\nWhat are the main differences between silent discos and traditional music festivals in terms of setup and experience?\nWhat impact can short meditation sessions have on daily stress?\nWhat is a common argument made by those concerned with the safety of nuclear energy facilities?\nWhat are some unique storytelling opportunities that virtual reality filmmaking offers which traditional media cannot provide?\nWhat are the main factors affecting soil porosity?\nWhat role does host susceptibility play in the transmission of viruses?\nWho is the bodhisattva associated with mercy, comfort, and forgiveness in Buddhist mythology?\nWhat role do markets play in determining prices in a capitalist economy?\nQ: What are the core beliefs or practices central to Hinduism that distinguish it from other major religions?\nWhat happens as a story unfolds in a place where everyone can become anyone else at will?\nWhat is the role of Mitochaal Genetic Makeup in determining a species traits?\nWhat are the key principles of modern military science?\nWhat part of book sculpting showcases the artist's skill in manipulating materials?\nIs there a connection between Shamanism and the local flora in Siberia?\nWhat role does the shutter speed play in low-light photography?\nWhat is your typical morning routine that significantly boosts your motivation and productivity for the day?\nWhat is a submersible used for in deep-sea exploration?\nQuestion: What are the primary methods of disease prevention used in public health?\nWhat is the role of mitochondria in a cell?\nWhat specific sensory memory of an event are you asking about in relation to cognition?\nHave you experienced ice climbing before, and if so, what unique challenges does it present compared to regular rock climbing?\nWhat are some proposed reforms for the modern justice system?\nWhat types of tax benefits might a homeowner receive if they decide to invest in solar panels for their property?\nWhat characteristics of dark matter make it different from ordinary matter?\nWhat are common ROM exercises recommended for knee injuries?\nQ: What is the principle behind a Foucault pendulum?\nWhat causes the geomagnetic reversal on Earth?\nWhat is one way neuroplasticity benefits individuals recovering from a stroke?\nWhat is the role of scarcity in determining market prices?\nWhat is the role of harmonic voices in enhancing the texture of polyphonic overtone singing?\nWhat effect does the interference of two closely spaced sounds at slightly different frequencies have on human perception of sound?\nWhat famous stories or myths are related to the theme of writing and authorship in ancient civilizations?\nWhat are some of the technological advancements that could lead to a solar punk future, emphasizing sustainability and renewable energy?\nWhat are some of the essential skills required for creative publishing a book?\nIn veterinary medicine, what are common preventive measures taken to control zoonotic diseases between pets and the general public?\nWhy do people experience heartbreak differently?\nWhat is the main challenge in creating stable anti-gravity systems that can levitate objects?\nWhat are the key components of the primary assessment in emergency medicine?\nWhat is the primary difference between bioacoustic monitoring and soundscape ecology?\nWhat historical event significantly impacted the formation of modern legal systems?\nWhat is the main venue where underground dance music gatherings typically take place?\nWhat is a unique challenge an urban artist might face when both sketching and painting in a plein air setting?\nWhat leadership challenges do Chief Executive Officers face when implementing digital transformation within a traditional company?\nWhat are some common challenges people face when they are speaking publicly, and how can they effectively overcome these hurdles?\nWhat impact might space tourism have on the choice of destinations for traditional commercial spaceflight businesses?\nQuestion: Can you explain Ohm's Law and its practical applications in electronic circuits?\nWhat urban details often inspire pleasure in plein air painters that should be focused on in a scene?\nCan you name two common sources of renewable energy?\nWhat are the potential impacts of widespread adoption of ultralight aviation technologies on urban air traffic management?\nWhat classic rock band from the 1970s had a significant impact on the development of rock music and continues to be popular among fans for their pioneering sound?\nWhy might someone choose temporary body painting or adhesive stickers over permanent tattoos?\nQ: What is the role of Gibbs free energy in predicting the spontaneity of a chemical reaction?\nWhat is the significance of the repetition of the word \"prayer\" in the concept of spirituality?\nQ: What is a common instrument often featured in pop music that accompanies vocals?\nQuestion: What is one main principle of humanism?\nWhat role did propaganda play before a significant event in history?\nWhat does a common food allergy trigger in the body?\nWhat ways can post-digital art incorporate elements of virtual reality to create immersive experiences for the viewer?\nWhat are the primary pollutants targeted by a typical flue gas desulfurization (FGD) system?\nCan you share a story from your community that embodies wisdom passed down through generations?\nWhat is an example of a use case for non-parametric statistics in a real-world scenario?\nWhat is the role of lithification in the rock cycle?\nWhat is an example of a technology that directly contributes to the development of smart cities by enabling communication and data exchange?\nWhat is a positive effect of practicing mindful eating on one's health?\nWhat is the primary function of nootropics in the body?\nWhat is the impact of soft power on smaller states' foreign policy decisions?\nWhat impact has religious freedom had on the development of ethical theories within major world religions?\nWhat role did slavery play in Ancient Egypt's economy?\nWhat are the main teachings about agency in Mormonism?\nWhat type of questions could I ask about living museums to understand their role in experiential history better?\nIs spiritual initiation always associated with complex symbolism and mysterious rituals in Gnostic traditions?\nWhat role does aerodynamics play in the stability of a spacecraft during re-entry?\nWho might Locke have been influenced by in his theory of mind?\nQ: What are the main agricultural systems used to cultivate maize?\nWhat type of physical challenges do climbers face when ice climbing a vertical wall?\nWhat are the main challenges faced by the telecommunications industry when rolling out a 5G network?\nQ: What is the primary function of the blood-brain barrier (BBB)?\nWhat role does backpropagation play in neural networks?\nWhat is the purpose of a dental crown in cavity fillings?\nWhat is the role of error detection and correction in computer networks?\nWhat are the main safety concerns associated with nuclear reactors?\nWhat role does improvisation play in the success of a live theater performance?\nWhat is the molar mass of glucose, C6H12O6?\nWhat is the common cause of hand numbness after a surgery?\nCan you explain the role of collective intelligence in a crowdsourcing project?\nWhat layers of the skin are examined in a skin biopsy for dermatopathology?\nWhy do slow cinema and minimalist filmmaking often evoke a stronger emotional response in the audience compared to high-paced films?\n",

# payload = {
#     "prompts": "What are the challenges of scaling quantum computing for practical applications?\nWhat is the role of blockchain in securing IoT devices?\nWhat is the difference between reinforcement learning and supervised learning in AI?\nWhat are common methods for identifying faults in electric vehicle batteries?\nHow does augmented reality enhance the retail shopping experience?\nWhat is the significance of edge computing in 5G networks?\nHow does gene editing using CRISPR technology work?\nWhat is the impact of renewable energy storage solutions on grid stability?\nWhat role do microservices play in modern software architecture?\nWhat are the main components of a successful digital marketing strategy?\nWhat is the role of phytoplankton in the global carbon cycle?\nHow does 3D printing revolutionize the manufacturing industry?\nWhat is a practical application of drone technology in agriculture?\nWhat are the ethical concerns surrounding AI decision-making?\nWhat is the primary difference between RNA and DNA vaccines?\nWhat are the factors influencing the adoption of electric vehicles globally?\nWhat impact does sleep quality have on cognitive performance?\nWhat is the process of desalination, and how is it applied in water-scarce regions?\nWhat role does cybersecurity play in the adoption of smart cities?\nHow does natural language processing power modern virtual assistants?\nWhat are the major challenges in designing sustainable packaging materials?\nWhat are the key differences between classical and quantum cryptography?\nWhat is the role of fungi in carbon sequestration?\nHow does neural architecture search improve AI model development?\nWhat is the role of enzymes in industrial chemical processes?\nWhat are some examples of using big data in personalized healthcare?\nWhat are the core principles of circular economy initiatives?\nWhat are the benefits of modular design in urban construction?\nWhat is the relationship between biodiversity and ecosystem stability?\nHow do autonomous vehicles use sensor fusion to make driving decisions?\nWhat are the primary obstacles in achieving nuclear fusion as a viable energy source?\nWhat is the role of virtual reality in educational training?\nHow does the Internet of Behaviors (IoB) influence consumer habits?\nWhat are the core components of a comprehensive disaster management system?\nWhat is the role of exoplanets in the search for extraterrestrial life?\nWhat are the implications of quantum entanglement in secure communication?\nWhat is the main purpose of machine vision in robotics?\nWhat challenges do researchers face in developing biodegradable plastics?\nWhat is the impact of climate change on polar ecosystems?\nWhat are some innovations in wearable technology for health monitoring?\nWhat is the role of swarm robotics in disaster response?\nHow does gamification enhance user engagement in e-learning platforms?\nWhat are the potential applications of DNA storage in data preservation?\nWhat is the significance of water footprint analysis in sustainable development?\nHow do predictive analytics improve supply chain management?\nWhat are the primary challenges of cybersecurity in autonomous systems?\nWhat is the role of nanotechnology in advanced drug delivery systems?\nHow do greenhouse gases contribute to global warming?\nWhat is the significance of dark matter in understanding the universe's composition?\nWhat are the principles of permaculture in sustainable agriculture?\nWhat are the applications of photogrammetry in environmental monitoring?\nWhat are the ethical considerations in using AI for hiring decisions?\nHow does cognitive behavioral therapy help in managing anxiety disorders?\nWhat are the benefits of regenerative braking in electric vehicles?\nWhat is the role of epigenetics in understanding hereditary diseases?\nHow does blockchain technology support decentralized finance (DeFi)?\nWhat are the differences between virtual reality and augmented reality in user experience?\nWhat are the potential risks and rewards of artificial general intelligence (AGI)?\nWhat is the role of telemetry in space exploration?\nWhat are the primary drivers of ocean acidification?\nHow does gene therapy work to treat genetic disorders?\nWhat are the advantages of solid-state batteries over traditional lithium-ion batteries?\nWhat role do lichens play in indicating air quality?\nWhat are the main challenges in integrating renewable energy into existing power grids?\nWhat is the impact of data visualization on decision-making processes?\nWhat are some examples of machine learning in predictive maintenance?\nWhat is the relationship between food security and sustainable agriculture?\nWhat are the core functionalities of a knowledge graph in data analytics?\nWhat are the benefits of robotic process automation in business operations?\nWhat are the challenges in training large-scale language models?\nWhat is the significance of the Turing Test in evaluating AI capabilities?\nHow does the concept of the metaverse influence virtual social interactions?\nWhat is the role of hydrogen fuel cells in achieving zero-emission goals?\nHow does biodiversity loss affect food production systems?\nWhat are the environmental impacts of fast fashion, and how can they be mitigated?\nWhat is the role of lithium recycling in the sustainability of battery technology?\nWhat are the principles of biomimicry in engineering design?\nWhat are the applications of quantum computing in cryptography?\nHow does photonics improve data transmission speeds in telecommunications?\nWhat is the role of probiotics in gut health?\nWhat are the key differences between traditional farming and vertical farming?\nWhat is the impact of microplastics on marine ecosystems?\nWhat role does design thinking play in solving complex business challenges?\nWhat is the significance of bioinformatics in personalized medicine?\nWhat are the implications of AI-driven automation for job markets?\nWhat are the main challenges in achieving sustainable fisheries?\nWhat role does adaptive learning play in modern education technology?\nWhat is the impact of 5G on industrial automation?\nHow does machine learning improve fraud detection in financial systems?\nWhat is the relationship between neural networks and deep learning?\nWhat are the key factors in developing carbon capture and storage technology?\nWhat are the environmental benefits of urban green roofs?\nWhat role does augmented analytics play in business intelligence?\nWhat are the principles of quantum supremacy in computational problems?\n",
    "temperature": "0.5",
    "top_p": "0.8",
    "max_tokens": "256",
    "seed": "3743504407",
}

#     "temperature": "0",
#     "top_p": "1.0",
#     "max_tokens": "256",
#     "seed": "3743504407",
# }

# # Specify the desired file path.  
# output_file_path = 'output_responses.txt'  

# # Run inference
# responses = solve_prompt(payload)

# with open(output_file_path, 'w') as file:  
#     for prompt, response in responses.items():  
#         file.write(f"Prompt: {prompt}\nResponse: {response}\n\n")  # Write to the file


# output_file_path = 'output_responses.txt'

# # Run inference
# responses = solve_prompt(payload)

# with open(output_file_path, 'w') as file:
#     for prompt, response in responses.items():
#         formatted_response = response.replace('\n', '\\n')  # Replace line breaks with \n
#         file.write(f'"{prompt}": "{formatted_response.strip()}",\n')  # Format output with indentation and quotes

import json

output_file_path = 'output_A40_vLLM_0.6.3.json'

# Run inference
responses = solve_prompt(payload)

# Prepare JSON format
json_output = [
    {
        "prompt": prompt,
        "response": response.replace('\n', '\\n')  # Replace line breaks with \n
    }
    for prompt, response in responses.items()
]

# Write to a JSON file
with open(output_file_path, 'w') as file:
    json.dump(json_output, file, indent=4)  # Pretty-print with 4 spaces for indentation
