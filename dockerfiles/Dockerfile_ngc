# Specify image
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Ensure apt is in non-interactive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
# Set TimeZone to KST
ENV TZ=Asia/Seoul

# Set working directory
WORKDIR /project
# Copy requirements
COPY ../requirements.txt /project/

# Expose port 80
EXPOSE 80

# Install dependencies
RUN \
    apt-get update &&\
    apt-get install -y --no-install-recommends \
        ca-certificates \
        ccache \
        cmake \
        curl \
        fonts-powerline \
        git \
        imagemagick \
        language-pack-en \
        # libfreetype6-dev \
        # libhdf5-serial-dev \
        # libzmq3-dev \
        # libjpeg-dev \
        # libpng-dev \
        # libsm6 \
        # libxext6 \
        # libxrender-dev \
        # pkg-config \
        # software-properties-common \
        ssh \
        sudo \
        tmux \
        tzdata \
        unzip \
        vim \
        wget \
        zsh


# Install ZSH(Z Shell)
RUN \
    chsh -s /bin/zsh &&\
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" &&\
    git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions &&\
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting &&\
    perl -pi -w -e 's/ZSH_THEME=.*/ZSH_THEME="agnoster"/g;' ~/.zshrc &&\
    perl -pi -w -e 's/plugins=.*/plugins=(git tmux ssh-agent zsh-autosuggestions zsh-syntax-highlighting)/g;' ~/.zshrc

# Prevent TimeZone mismatch error
RUN update-locale


# Install Miniconda with Root perimission
ENV LANG C.UTF-8
RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm /tmp/miniconda.sh
RUN conda update -y conda

# Setup SSH configuration
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config

# Install requirements
RUN pip install -r requirements.txt
# NOTE: Recent pip(> 20.3) has backtracking runtime issue
# (See https://pip.pypa.io/en/stable/topics/dependency-resolution/#possible-ways-to-reduce-backtracking)
# RUN \
#     pip install -U pip==20.3 &&\
#     pip install --use-deprecated=legacy-resolver -r requirements.txt

# Cleanup Cache files
RUN \
    apt-get clean &&\
    apt-get autoclean &&\
    apt-get autoremove -y &&\
    rm -rf /var/lib/cache/* &&\
    rm -rf /var/lib/log/*