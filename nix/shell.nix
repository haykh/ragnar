{
  pkgs ? import <nixpkgs> { },
  openmp ? false,
  gpu ? "NONE",
  arch ? "NATIVE",
}:

let
  gpuUpper = pkgs.lib.toUpper gpu;
  archUpper = pkgs.lib.toUpper arch;
  name = "ragnar-dev";
  kokkosPkg = (
    pkgs.callPackage ./kokkos.nix {
      inherit pkgs;
      inherit openmp;
      arch = archUpper;
      gpu = gpuUpper;
    }
  );
  envVars = {
    compiler = rec {
      NONE = {
        CXX = "g++";
        CC = "gcc";
      };
      HIP = {
        CXX = "hipcc";
        CC = "hipcc";
      };
      CUDA = NONE;
    };
    kokkos = {
      HIP = {
        Kokkos_ENABLE_HIP = "ON";
      };
      CUDA = {
        Kokkos_ENABLE_CUDA = "ON";
      };
      NONE = (if openmp then { Kokkos_ENABLE_OPENMP = "ON"; } else { });
    };
  };
in
pkgs.mkShell {
  name = "${name}-env";
  nativeBuildInputs = with pkgs; [
    zlib
    cmake

    clang-tools

    kokkosPkg
    hdf5
    highfive
    toml11

    python312
    python312Packages.jupyter

    cmake-format
    cmake-lint
    neocmakelsp
    black
    pyright
    taplo
    vscode-langservers-extracted
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ]);

  shellHook =
    ''
      BLUE='\033[0;34m'
      NC='\033[0m'

      echo "following environment variables are set:"
    ''
    + pkgs.lib.concatStringsSep "" (
      pkgs.lib.mapAttrsToList (
        category: vars:
        pkgs.lib.concatStringsSep "" (
          pkgs.lib.mapAttrsToList (name: value: ''
            export ${name}=${value}
            echo -e "  ''\${BLUE}${name}''\${NC}=${value}"
          '') vars.${gpuUpper}
        )
      ) envVars
    )
    + ''
      echo ""
      echo -e "${name} nix-shell activated"
    '';

}
