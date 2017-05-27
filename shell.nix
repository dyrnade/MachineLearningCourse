(
let
  pkgs = import <nixpkgs> {
    config = {
      packageOverrides = pkgs: {
        python27 = let
          packageOverrides = self: super: {
            matplotlib = super.matplotlib.override  { enableGtk2 = true; } ;
          };
        in pkgs.python27.override {
          inherit packageOverrides;
        };
      };
    };
  };
in pkgs.python27.buildEnv.override { 
  extraLibs = with pkgs.python27Packages; [matplotlib numpy ipython];
  ignoreCollisions = true;
#pkgs.python27.withPackages (ps: [ps.matplotlib ps.numpy ps.ipython])
}).env
