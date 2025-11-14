{pkgs}: {
  deps = [
    pkgs.unzip
    pkgs.libGLU
    pkgs.libGL
    pkgs.postgresql
    pkgs.openssl
  ];
}
