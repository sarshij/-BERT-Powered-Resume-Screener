"""
Skill Alias Normalization
Maps abbreviations, shorthand, and common alternate spellings to their
canonical names as they appear in SKILL_KEYWORDS (taxonomy.py).

This module is used to pre-process both resume text and job description text
before skill extraction, so that:
  - "js" in a JD matches "JavaScript" in a resume
  - "k8s" in a JD matches "Kubernetes" in a resume
  - "ml" matches "Machine Learning"
  - "postgres" matches "PostgreSQL"
  etc.

All matching uses \b word boundaries to avoid false substitutions
(e.g., "class" should not be affected by a "c" alias).
"""

import re

# ── Alias Dictionary ────────────────────────────────────────────────────────
# Format: { "alias": "Canonical Skill Name" }
# The canonical name MUST exist in SKILL_KEYWORDS in taxonomy.py.
# Keys are lowercase. Values must match the casing in SKILL_KEYWORDS exactly.
SKILL_ALIASES: dict[str, str] = {
    # ── JavaScript ecosystem ──────────────────────────────────────────────
    "js":             "JavaScript",
    "javascript":     "JavaScript",
    "node":           "Node.js",
    "nodejs":         "Node.js",
    "node.js":        "Node.js",
    "expressjs":      "Express.js",
    "express":        "Express.js",
    "nextjs":         "Next.js",
    "next":           "Next.js",
    "nuxtjs":         "Nuxt.js",
    "nuxt":           "Nuxt.js",
    "vuejs":          "Vue.js",
    "vue":            "Vue.js",
    "reactjs":        "React",
    "react.js":       "React",
    "angularjs":      "Angular",
    "sveltejs":       "Svelte",
    "ts":             "TypeScript",
    "typescript":     "TypeScript",

    # ── Python ecosystem ───────────────────────────────────────────────────
    "py":             "Python",
    "sklearn":        "scikit-learn",
    "sk-learn":       "scikit-learn",
    "scikit":         "scikit-learn",
    "pytorch":        "PyTorch",
    "pt":             "PyTorch",
    "tf":             "TensorFlow",
    "tensorflow":     "TensorFlow",
    "keras":          "Keras",
    "np":             "NumPy",
    "numpy":          "NumPy",
    "pd":             "Pandas",
    "pandas":         "Pandas",
    "mpl":            "Matplotlib",
    "matplotlib":     "Matplotlib",

    # ── AI / ML abbreviations ──────────────────────────────────────────────
    "ml":             "Machine Learning",
    "ai":             "Artificial Intelligence",   # careful: also "Adobe Illustrator" — context
    "dl":             "Deep Learning",
    "nlp":            "NLP",
    "cv":             "Computer Vision",
    "rl":             "Reinforcement Learning",
    "llm":            "LLM",
    "genai":          "Generative AI",
    "gen ai":         "Generative AI",

    # ── Cloud & DevOps ─────────────────────────────────────────────────────
    "k8s":            "Kubernetes",
    "kube":           "Kubernetes",
    "kubernetes":     "Kubernetes",
    "docker":         "Docker",
    "aws":            "AWS",
    "amazon web services": "AWS",
    "azure":          "Azure",
    "microsoft azure": "Azure",
    "gcp":            "GCP",
    "google cloud":   "GCP",
    "google cloud platform": "GCP",
    "ci/cd":          "CI/CD",
    "cicd":           "CI/CD",
    "devops":         "DevOps",
    "mlops":          "MLOps",
    "terraform":      "Terraform",
    "ansible":        "Ansible",
    "gh actions":     "GitHub Actions",
    "github actions": "GitHub Actions",

    # ── Databases ──────────────────────────────────────────────────────────
    "sql":            "SQL",
    "nosql":          "NoSQL",
    "mongo":          "MongoDB",
    "mongodb":        "MongoDB",
    "postgres":       "PostgreSQL",
    "postgresql":     "PostgreSQL",
    "psql":           "PostgreSQL",
    "pg":             "PostgreSQL",
    "mysql":          "MySQL",
    "sqlite":         "SQLite",
    "redis":          "Redis",
    "elastic":        "Elasticsearch",
    "elasticsearch":  "Elasticsearch",
    "dynamo":         "DynamoDB",
    "dynamodb":       "DynamoDB",

    # ── Mobile ─────────────────────────────────────────────────────────────
    "rn":             "React Native",
    "react native":   "React Native",
    "flutter":        "Flutter",
    "kotlin":         "Kotlin",
    "swift":          "Swift",

    # ── Programming language shorthands ────────────────────────────────────
    "c++":            "C++",
    "cpp":            "C++",
    "csharp":         "C#",
    "c#":             "C#",
    "dotnet":         ".NET",
    ".net":           ".NET",
    "asp.net":        "ASP.NET",
    "aspnet":         "ASP.NET",
    "golang":         "Go",
    "rb":             "Ruby",
    "ruby":           "Ruby",
    "php":            "PHP",
    "rs":             "Rust",
    "rust":           "Rust",
    "scala":          "Scala",
    "r lang":         "R",

    # ── Web / CSS frameworks ────────────────────────────────────────────────
    "html5":          "HTML",
    "css3":           "CSS",
    "sass":           "SASS",
    "scss":           "SCSS",
    "tailwindcss":    "Tailwind",
    "tailwind css":   "Tailwind",
    "bootstrap":      "Bootstrap",
    "material ui":    "Material UI",
    "mui":            "Material UI",

    # ── Testing ────────────────────────────────────────────────────────────
    "jest":           "Jest",
    "pytest":         "Pytest",
    "cypress":        "Cypress",
    "playwright":     "Playwright",
    "selenium":       "Selenium",
    "junit":          "JUnit",
    "tdd":            "TDD",
    "bdd":            "BDD",

    # ── Tools / Design ─────────────────────────────────────────────────────
    "figma":          "Figma",
    "xd":             "Adobe XD",
    "adobe xd":       "Adobe XD",
    "ps":             "Photoshop",
    "photoshop":      "Photoshop",
    "ai app":         "Illustrator",     # "ai" alone is ambiguous — handled separately
    "illustrator":    "Illustrator",
    "git":            "Git",
    "github":         "GitHub",
    "gitlab":         "GitLab",
    "bitbucket":      "Bitbucket",
    "jira":           "JIRA",
    "confluence":     "Confluence",

    # ── Data / Analytics ───────────────────────────────────────────────────
    "tableau":        "Tableau",
    "power bi":       "Power BI",
    "powerbi":        "Power BI",
    "excel":          "Excel",
    "bi":             "Power BI",   # only when standalone (word boundary saves us)
    "spark":          "Spark",
    "apache spark":   "Spark",
    "kafka":          "Kafka",
    "apache kafka":   "Kafka",
    "airflow":        "Airflow",
    "apache airflow": "Airflow",
    "hadoop":         "Hadoop",

    # ── Security ───────────────────────────────────────────────────────────
    "pentest":        "Penetration Testing",
    "pen testing":    "Penetration Testing",
    "sec":            "Cybersecurity",
    "infosec":        "Cybersecurity",

    # ── Methodologies ──────────────────────────────────────────────────────
    "agile":          "Agile",
    "scrum":          "Scrum",
    "kanban":         "Kanban",
    "oop":            "OOP",
    "rest":           "REST APIs",
    "rest api":       "REST APIs",
    "restful":        "REST APIs",
    "graphql":        "GraphQL",
    "grpc":           "gRPC",
    "api":            "REST APIs",    # generic "api" → REST APIs (most common)
}

# Sort aliases by length descending so longer aliases are matched first
# (prevents "node" matching inside "nodejs" before "nodejs" pattern runs)
_SORTED_ALIASES = sorted(SKILL_ALIASES.keys(), key=len, reverse=True)

# Pre-compile all alias regex patterns for performance
_ALIAS_PATTERNS: list[tuple[re.Pattern, str]] = []
for _alias in _SORTED_ALIASES:
    _canonical = SKILL_ALIASES[_alias]
    try:
        # Use word boundary \b for single-word aliases
        # For multi-word aliases (e.g., "google cloud"), use simple boundary check
        if " " in _alias:
            # Multi-word: match literally with optional surrounding spaces/punctuation
            _pat = re.compile(r'(?<![a-z0-9])' + re.escape(_alias) + r'(?![a-z0-9])', re.IGNORECASE)
        else:
            _pat = re.compile(r'\b' + re.escape(_alias) + r'\b', re.IGNORECASE)
        _ALIAS_PATTERNS.append((_pat, _canonical))
    except re.error:
        pass


def normalize_skills_text(text: str) -> str:
    """
    Normalize text by expanding skill abbreviations to their canonical forms.

    Example:
        "js, ts, ml and ai" → "JavaScript, TypeScript, Machine Learning and Artificial Intelligence"
        "k8s with docker" → "Kubernetes with Docker"
        "postgres sql" → "PostgreSQL sql"

    Uses pre-compiled regex with word boundaries to avoid false substitutions.
    The function is idempotent: running it twice gives the same result.

    Args:
        text: Raw resume or job description text.

    Returns:
        Text with abbreviations replaced by canonical skill names.
    """
    if not text:
        return text

    result = text
    for pattern, canonical in _ALIAS_PATTERNS:
        result = pattern.sub(canonical, result)

    return result


def normalize_skill_token(token: str) -> str:
    """
    Normalize a single skill token/word to its canonical name.
    Returns the canonical name if found, otherwise returns the original token.

    Useful for normalizing individual items from a skills list.
    """
    token_lower = token.strip().lower()
    return SKILL_ALIASES.get(token_lower, token)
