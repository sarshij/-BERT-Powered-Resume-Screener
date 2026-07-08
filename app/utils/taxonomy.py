"""
Skills taxonomy, job categories, and generic phrases for resume analysis.
Comprehensive skill list covering all major tech stacks, tools, and soft skills.
"""

# ── Skill Keywords ──────────────────────────────────────────────────────────
# BUG 6 FIX: Added many missing common skills (HTML, CSS, PHP, Ruby, etc.)
# that were causing skill overlap to return near-zero for web dev resumes.
SKILL_KEYWORDS = [
    # ── Programming Languages ──
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C",
    "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Dart", "R",
    "Scala", "Perl", "Lua", "Haskell", "Elixir", "Clojure",
    "Assembly", "MATLAB", "Julia", "Objective-C", "Visual Basic",
    "COBOL", "Fortran", "Groovy",

    # ── Web Frontend ──
    "HTML", "CSS", "SASS", "SCSS", "Less",
    "React", "Vue.js", "Angular", "Svelte", "Next.js", "Nuxt.js",
    "jQuery", "Bootstrap", "Tailwind", "Material UI",
    "Webpack", "Vite", "Babel", "ESLint", "Prettier",
    "Redux", "Zustand", "MobX",
    "Responsive Design", "Web Accessibility", "PWA",

    # ── Web Backend ──
    "Node.js", "Express.js", "Django", "Flask", "FastAPI",
    "Spring Boot", "Spring", "Laravel", "Ruby on Rails",
    "ASP.NET", ".NET", "Gin", "Fiber",
    "REST APIs", "GraphQL", "gRPC", "WebSocket",
    "Microservices", "Monolith", "Serverless",

    # ── Databases ──
    "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "SQLite",
    "Redis", "Cassandra", "DynamoDB", "Firebase",
    "Elasticsearch", "Neo4j", "CouchDB", "MariaDB",
    "Oracle Database", "Microsoft SQL Server", "Supabase",

    # ── Cloud & DevOps ──
    "AWS", "Azure", "GCP", "Docker", "Kubernetes",
    "Terraform", "Ansible", "Pulumi", "CloudFormation",
    "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI",
    "CI/CD", "Linux", "Bash", "Shell", "PowerShell",
    "Nginx", "Apache", "Heroku", "Vercel", "Netlify",
    "DigitalOcean", "Vagrant", "Prometheus", "Grafana",
    "ELK Stack", "Datadog", "New Relic",

    # ── AI / ML / Data Science ──
    "TensorFlow", "PyTorch", "Keras", "scikit-learn",
    "NumPy", "Pandas", "Matplotlib", "Seaborn",
    "Machine Learning", "Deep Learning", "NLP",
    "Computer Vision", "Reinforcement Learning",
    "BERT", "GPT", "LLM", "RAG", "LangChain",
    "Hugging Face", "OpenCV", "YOLO", "Transformers",
    "XGBoost", "LightGBM", "Random Forest", "SVM",
    "Neural Networks", "CNN", "RNN", "LSTM", "GAN",
    "Feature Engineering", "Model Deployment",

    # ── Data Engineering ──
    "Spark", "Hadoop", "Kafka", "Airflow", "ETL",
    "Snowflake", "Databricks", "BigQuery", "Redshift",
    "Data Pipeline", "Data Warehouse", "Data Lake",
    "dbt", "Prefect", "Luigi", "Celery",

    # ── Version Control ──
    "Git", "GitHub", "GitLab", "Bitbucket", "SVN",

    # ── Mobile Development ──
    "React Native", "Flutter", "SwiftUI", "Jetpack Compose",
    "Xamarin", "Ionic", "Cordova", "Android", "iOS",

    # ── Testing ──
    "Unit Testing", "Integration Testing", "E2E Testing",
    "Jest", "Pytest", "Selenium", "Cypress", "Playwright",
    "JUnit", "Mocha", "Chai", "TestNG",
    "TDD", "BDD", "Code Review",

    # ── Security ──
    "Cybersecurity", "Network Security", "Penetration Testing",
    "OWASP", "SSL/TLS", "OAuth", "JWT",
    "Encryption", "Firewall", "SIEM", "SOC",

    # ── Design & UI/UX ──
    "Figma", "Sketch", "Adobe XD", "InVision",
    "Photoshop", "Illustrator", "After Effects",
    "UI Design", "UX Design", "Wireframing", "Prototyping",
    "User Research", "A/B Testing",

    # ── Project Management & Methodology ──
    "Agile", "Scrum", "Kanban", "JIRA", "Confluence",
    "Trello", "Asana", "Monday.com", "Notion",
    "Waterfall", "Lean", "Six Sigma",

    # ── Engineering Tools ──
    "AutoCAD", "SolidWorks", "MATLAB", "Simulink",
    "FPGA", "Embedded C", "RTOS", "Arduino", "Raspberry Pi",
    "PCB Design", "VHDL", "Verilog",

    # ── Blockchain & Web3 ──
    "Blockchain", "Solidity", "Web3", "Ethereum",
    "Smart Contracts", "DeFi", "NFT", "Hardhat",

    # ── Analytics & BI ──
    "Tableau", "Power BI", "Excel", "Google Analytics",
    "Looker", "Mixpanel", "Amplitude",
    "Data Visualization", "Statistical Analysis",
    "Hypothesis Testing", "Regression Analysis",

    # ── Marketing & Content ──
    "SEO", "SEM", "Digital Marketing", "Content Strategy",
    "Google Ads", "Facebook Ads", "Social Media Marketing",
    "Email Marketing", "Content Marketing", "Copywriting",

    # ── Business & Finance ──
    "Accounting", "Financial Analysis", "Risk Management",
    "Business Intelligence", "CRM", "Salesforce", "SAP",
    "ERP", "Supply Chain Management",

    # ── Soft Skills ──
    "Project Management", "Leadership", "Team Management",
    "Communication", "Problem Solving", "Critical Thinking",
    "Time Management", "Collaboration", "Mentoring",
    "Presentation Skills", "Negotiation", "Decision Making",

    # ── Architecture & Concepts ──
    "OOP", "Data Structures", "Algorithms", "System Design",
    "DevOps", "MLOps", "DataOps", "Data Engineering",
    "Cloud Computing", "Edge Computing", "IoT",
    "API Design", "Design Patterns", "Clean Code",
    "SOLID Principles", "Domain-Driven Design",

    # ── Emerging Tech ──
    "Generative AI", "Prompt Engineering", "AI Agents",
    "Quantum Computing", "AR/VR", "5G",
    "Robotics", "Computer Graphics", "Game Development",
    "Unity", "Unreal Engine",
]

# ── Job Categories ──────────────────────────────────────────────────────────
JOB_CATEGORIES = {
    "Engineering": ["Software Engineer", "Backend Developer", "Frontend Developer",
                    "Full-Stack Developer", "DevOps Engineer", "Cloud Engineer",
                    "Data Engineer", "MLOps Engineer", "Site Reliability Engineer",
                    "Web Developer", "Software Developer", "Systems Engineer"],
    "AI_ML": ["Machine Learning Engineer", "Data Scientist", "AI Engineer",
              "Research Scientist", "Computer Vision Engineer", "NLP Engineer",
              "Deep Learning Engineer", "AI Researcher"],
    "Management": ["Product Manager", "Project Manager", "Engineering Manager",
                   "Technical Lead", "Team Lead", "Scrum Master",
                   "Program Manager", "CTO", "VP Engineering"],
    "Design": ["UI/UX Designer", "Product Designer", "Graphic Designer",
               "Visual Designer", "Interaction Designer", "UX Researcher"],
    "Security": ["Cybersecurity Engineer", "Security Analyst", "Penetration Tester",
                 "Security Architect", "SOC Analyst", "Security Engineer"],
    "Data": ["Data Analyst", "Data Engineer", "Data Scientist", "Business Analyst",
             "Database Administrator", "BI Developer", "Analytics Engineer"],
    "IT": ["System Administrator", "Network Engineer", "IT Support",
           "Database Administrator", "Cloud Administrator", "Help Desk"],
    "Finance": ["Financial Analyst", "Accountant", "Auditor", "Finance Manager",
                "Risk Analyst", "Investment Analyst"],
    "HR": ["Human Resources Manager", "HR Coordinator", "Recruiter",
           "Talent Acquisition Specialist", "HR Business Partner"],
    "Marketing": ["Marketing Manager", "Digital Marketing Specialist", "SEO Specialist",
                  "Content Strategist", "Growth Hacker", "Brand Manager"],
    "Mobile": ["Android Developer", "iOS Developer", "Mobile Developer",
               "Flutter Developer", "React Native Developer"],
    "QA": ["QA Engineer", "Test Engineer", "SDET", "Quality Assurance Analyst",
           "Automation Tester"],
}

# ── Generic Phrases ─────────────────────────────────────────────────────────
# Cliché / buzzword phrases commonly found in fake or low-quality resumes
GENERIC_PHRASES = [
    "results-driven", "team player", "think outside the box", "go-getter",
    "self-starter", "detail-oriented", "highly motivated", "passionate about",
    "hardworking", "dedicated professional", "proven track record",
    "excellent communication skills", "strong work ethic", "creative thinker",
    "problem solver", "people person", "quick learner", "strategic thinker",
    "customer-focused", "results oriented", "hit the ground running",
    "synergy", "leverage", "utilize", "dynamic", "innovative",
    "forward-thinking", "take initiative", "work well under pressure",
    "multitasker", "team-oriented", "goal-oriented", "action-oriented",
    "deep dive", "drill down", "breadth and depth", "best in class",
    "world-class", "cutting-edge", "state-of-the-art", "bleeding edge",
    "think strategically", "drive results", "make an impact",
    "deliver value", "add value", "business acumen",
    "stakeholder management", "cross-functional", "end-to-end",
    "holistic approach", "out of the box", "blue sky thinking",
    "win-win", "low hanging fruit", "move the needle",
    "paradigm shift", "streamline", "optimize", "synergize",
    "circle back", "touch base", "align", "bandwidth",
    "key player", "game changer", "thought leader",
    "best of breed", "enterprise-grade", "scalable",
    "sought after", "in high demand", "ninja", "guru", "rockstar",
]
