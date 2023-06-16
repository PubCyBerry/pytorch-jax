ARG NGC_YEAR=23
ARG NGC_MONTH=05
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:${NGC_YEAR}.${NGC_MONTH}-py3

# Specify image
FROM ${BASE_IMAGE}

# Define user arguments
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG USER_GROUP=user
ARG PASSWD=ubuntu

# Ensure apt is in non-interactive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
# Set TimeZone to KST
ENV TZ=Asia/Seoul

# Set working directory
ENV PROJECT_ROOT=/project
WORKDIR ${PROJECT_ROOT}
# Copy requirements.txt / environments.yaml
COPY ../requirements.txt /project/
COPY ../environments.yaml /project/

# Expose ports
# -- default ports --
# ssh : 22
# HTTP : 80
# mysql : 3306
# (See: https://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers)
EXPOSE 22 80 2224 3306

# Install dependencies
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        dialog \
        fonts-powerline \
        git \
        imagemagick \
        language-pack-en \
        # libhdf5-serial-dev \
        # libzmq3-dev \
        # libjpeg-dev \
        # libpng-dev \
        # libsm6 \
        # libxext6 \
        # libxrender-dev \
        # pkg-config \
        # software-properties-common \
        openssh-server \
        ssh \
        sudo \
        tmux \
        tzdata \
        unzip \
        vim \
        wget \
        zsh &&\
    apt-get autoremove -y &&\
    apt-get clean -y

# Setup tmux
RUN git clone https://github.com/gpakosz/.tmux.git ~/.tmux &&\
    ln -s -f ~/.tmux/.tmux.conf &&\
    cp ~/.tmux/.tmux.conf.local ~/.tmux.conf.local &&\
    echo "set-option -g default-shell /bin/zsh" >~/.tmux.conf.local

# Setup fzf
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf &&\
    ~/.fzf/install --all

# Install ZSH(Z Shell)
RUN chsh -s /bin/zsh &&\
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" &&\
    git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions &&\
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting &&\
    perl -pi -w -e 's/ZSH_THEME=.*/ZSH_THEME="agnoster"/g;' ~/.zshrc &&\
    perl -pi -w -e 's/plugins=.*/plugins=(git tmux fzf ssh-agent zsh-autosuggestions zsh-syntax-highlighting)/g;' ~/.zshrc

# Prevent TimeZone mismatch error
RUN update-locale

# Setup SSH configuration
RUN echo "PasswordAuthentication yes" >>/etc/ssh/sshd_config &&\
    echo "PermitEmptyPasswords yes" >>/etc/ssh/sshd_config &&\
    echo "UsePAM no" >>/etc/ssh/sshd_config

# Install requirements
RUN pip install -r requirements.txt

# Set git config
RUN git config --global init.defaultBranch "main" &&\
    git config --global core.editor "vim" &&\
    git config --global --add safe.directory ${PROJECT_ROOT}

# Reset apt to be interactive
ENV DEBIAN_FRONTEND=dialog

# # Create user with password-free `sudo` permissions.
# RUN groupadd -f -g ${USER_GID} ${USER_GROUP} && \
#     useradd --shell $(which zsh) --create-home -u ${USER_UID} -g ${USER_GROUP} \
#         -p $(openssl passwd -1 ${PASSWD}) ${USERNAME} && \
#     echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# # Set default user
# USER $USERNAME

# Cleanup Cache files, non-existing symbolic link
RUN apt-get clean &&\
    apt-get autoclean &&\
    apt-get autoremove -y &&\
    rm -rf /var/lib/cache/* &&\
    rm -rf /var/lib/log/* &&\
    rm -f /usr/local/{libtbbmalloc_proxy.so.2,libtbbbind_2_0.so.3,libtbb.so.12,libtbbmalloc.so.2,libtbbbind_2_5.so.3,libtbbbind.so.3}

CMD [ "/bin/zsh" ]
