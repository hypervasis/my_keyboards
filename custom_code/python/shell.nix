{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {


  # This fixes an error of:
  # ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory
  # while running matplotlib
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/";

  buildInputs = [
    pkgs.python3
    pkgs.poetry
    pkgs.sqlite-utils
    # pkgs.postgresql # required for psycopg2
    # pkgs.python311Packages.invoke
    # pkgs.python311Packages.rich
    # pkgs.python311Packages.gitpython
    # pkgs.python311Packages.semver
    # pkgs.tilt # brings up local dev env
  ];


}
