# === SETUP.PY ===

# Save this as: setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:

    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:

    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(

    name="bizwiz-enhanced",

    version="2.2.0",

    author="Nick Bergeland",

    author_email="nnbergeland@gmail.comm",

    description="Enhanced Multi-City Commercial Location Analysis System",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/nbergeland/BIZWIZV2.1",

    packages=find_packages(),

    classifiers=[

        "Development Status :: 4 - Beta",

        "Intended Audience :: Developers",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

        "Programming Language :: Python :: 3",

        "Programming Language :: Python :: 3.8",

        "Programming Language :: Python :: 3.9",

        "Programming Language :: Python :: 3.10",

        "Programming Language :: Python :: 3.11",

    ],

    python_requires=">=3.8",

    install_requires=requirements,

    extras_require={

        "dev": [

            "pytest>=7.0.0",

            "black>=22.0.0",

            "flake8>=5.0.0",

            "mypy>=0.971",

        ],

        "geo": [

            "geopandas>=0.11.0",

            "shapely>=1.8.0",

            "folium>=0.12.0",

        ]

    },

    entry_points={

        "console_scripts": [

            "bizwiz-collect=enhanced_data_collection:main",

            "bizwiz-app=enhanced_visualization_app:main",

        ],

    },

)

