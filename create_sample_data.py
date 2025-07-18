import os
from pathlib import Path

def create_sample_data():
    """Create sample career data files for testing"""
    
    sample_data_dir = Path("sample_data")
    sample_data_dir.mkdir(exist_ok=True)
    
    # Sample career data files
    sample_files = {
        "software_engineering_career.txt": """
Software Engineering Career Guide

Career Overview:
Software engineering is a field that involves designing, developing, testing, and maintaining software applications and systems. It combines computer science principles with engineering practices to create efficient, scalable, and reliable software solutions.

Eligibility & Path:
- Bachelor's degree in Computer Science, Software Engineering, or related field
- Strong programming skills in languages like Python, Java, C++, JavaScript
- Understanding of software development methodologies (Agile, Scrum)
- Knowledge of data structures, algorithms, and system design

Important Exams:
- GATE (Graduate Aptitude Test in Engineering)
- JEE (Joint Entrance Examination)
- BITSAT (Birla Institute of Technology and Science Admission Test)
- VITEEE (VIT Engineering Entrance Examination)

Free Courses:
Coursera:
- Python for Everybody Specialization
- Full Stack Web Development Specialization
- Google IT Support Professional Certificate

SWAYAM:
- Programming in Java
- Data Structures and Algorithms
- Software Engineering

Best Colleges in India:
- Indian Institute of Technology (IIT) Delhi
- Indian Institute of Technology (IIT) Bombay
- Indian Institute of Technology (IIT) Madras
- Indian Institute of Science (IISc) Bangalore
- Delhi Technological University (DTU)

Best Colleges Abroad:
- Massachusetts Institute of Technology (MIT), USA
- Stanford University, USA
- Carnegie Mellon University, USA
- University of Oxford, UK
- ETH Zurich, Switzerland

Job Profiles:
- Software Developer
- Full Stack Developer
- Backend Developer
- Frontend Developer
- DevOps Engineer
- Software Architect
- Technical Lead

Salary Expectations:
- Entry Level (0-2 years): ₹3-8 LPA
- Mid Level (2-5 years): ₹8-18 LPA
- Senior Level (5-10 years): ₹18-40 LPA
- Lead/Architect (10+ years): ₹40+ LPA
""",
        
        "data_science_career.txt": """
Data Science Career Guide

Career Overview:
Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, machine learning, and domain expertise.

Eligibility & Path:
- Bachelor's degree in Computer Science, Statistics, Mathematics, or related field
- Strong programming skills in Python, R, SQL
- Knowledge of statistics, machine learning, and data visualization
- Understanding of big data technologies (Hadoop, Spark)

Important Exams:
- GATE (Computer Science/Mathematics)
- JAM (Joint Admission Test for M.Sc.)
- TIFR (Tata Institute of Fundamental Research)
- ISI Admission Test

Free Courses:
Coursera:
- IBM Data Science Professional Certificate
- Google Data Analytics Professional Certificate
- Machine Learning by Andrew Ng

SWAYAM:
- Introduction to Data Science
- Statistical Methods for Data Science
- Machine Learning Foundations

Best Colleges in India:
- Indian Statistical Institute (ISI)
- Indian Institute of Technology (IIT) Delhi
- Indian Institute of Technology (IIT) Bombay
- Chennai Mathematical Institute (CMI)
- Indian Institute of Science (IISc) Bangalore

Best Colleges Abroad:
- Stanford University, USA
- Massachusetts Institute of Technology (MIT), USA
- Harvard University, USA
- University of California Berkeley, USA
- University of Oxford, UK

Job Profiles:
- Data Scientist
- Machine Learning Engineer
- Data Analyst
- Business Intelligence Analyst
- Research Scientist
- Data Engineer
- AI/ML Consultant

Salary Expectations:
- Entry Level (0-2 years): ₹4-10 LPA
- Mid Level (2-5 years): ₹10-20 LPA
- Senior Level (5-10 years): ₹20-45 LPA
- Lead/Principal (10+ years): ₹45+ LPA
""",
        
        "machine_learning_career.txt": """
Machine Learning Career Guide

Career Overview:
Machine Learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to improve their performance on tasks through experience without being explicitly programmed.

Eligibility & Path:
- Bachelor's degree in Computer Science, Mathematics, Statistics, or Engineering
- Strong programming skills in Python, R, Java, C++
- Knowledge of linear algebra, calculus, statistics, and probability
- Understanding of ML algorithms and deep learning frameworks

Important Exams:
- GATE (Computer Science/Mathematics)
- GRE (for international programs)
- TOEFL/IELTS (for international programs)
- Company-specific technical interviews

Free Courses:
Coursera:
- Machine Learning Specialization by Andrew Ng
- Deep Learning Specialization
- TensorFlow Developer Professional Certificate

SWAYAM:
- Introduction to Machine Learning
- Deep Learning
- Artificial Intelligence

Best Colleges in India:
- Indian Institute of Technology (IIT) Delhi
- Indian Institute of Technology (IIT) Bombay
- Indian Institute of Technology (IIT) Madras
- Indian Institute of Science (IISc) Bangalore
- International Institute of Information Technology (IIIT) Hyderabad

Best Colleges Abroad:
- Stanford University, USA
- Massachusetts Institute of Technology (MIT), USA
- Carnegie Mellon University, USA
- University of California Berkeley, USA
- University of Toronto, Canada

Job Profiles:
- Machine Learning Engineer
- AI Research Scientist
- Deep Learning Engineer
- Computer Vision Engineer
- Natural Language Processing Engineer
- MLOps Engineer
- AI Product Manager

Salary Expectations:
- Entry Level (0-2 years): ₹5-12 LPA
- Mid Level (2-5 years): ₹12-25 LPA
- Senior Level (5-10 years): ₹25-50 LPA
- Lead/Principal (10+ years): ₹50+ LPA
"""
    }
    
    # Write sample files
    for filename, content in sample_files.items():
        with open(sample_data_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"✅ Created {len(sample_files)} sample data files in {sample_data_dir}")
    print("Sample files created:")
    for filename in sample_files.keys():
        print(f"  - {filename}")

if __name__ == "__main__":
    create_sample_data() 