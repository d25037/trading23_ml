FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app
COPY . .

RUN apt update && apt install -y vim zsh git curl bat fzf build-essential procps file

# set up bat
RUN mkdir -p ~/.local/bin
RUN ln -s /usr/bin/batcat ~/.local/bin/bat

# install rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# install starship
RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes

# install zoxide
RUN cargo install zoxide --locked

# install eza
RUN cargo install eza

# # install brew
# RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" | sh -s -- --yes
# RUN echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> $HOME/.profile
# RUN eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

# nvim
# RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz
# RUN rm -rf /opt/nvim
# RUN tar -C /opt -xzf nvim-linux64.tar.gz
# RUN git clone --depth 1 https://github.com/AstroNvim/AstroNvim ~/.config/nvim

RUN ln -s /app/.zshrc /root/.zshrc
RUN chsh -s /usr/bin/zsh