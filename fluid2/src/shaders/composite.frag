#version 330 core
in vec2 uv;
out vec4 out_color;
uniform sampler2D scene;
uniform vec2 pixel;
void main() {
    vec3 color=texture(scene,uv).rgb;
    vec3 bloom=vec3(0);
    for(int i=0;i<12;i++) {
        float angle=float(i)*2.3999632;
        vec2 offset=vec2(cos(angle),sin(angle))*sqrt(float(i)+1.0)*3.0*pixel;
        bloom+=max(texture(scene,uv+offset).rgb-vec3(0.32),vec3(0));
    }
    color+=bloom*(0.28/12.0);
    color=color/(0.7+color);
    out_color=vec4(pow(color,vec3(1.0/2.2)),1.0);
}
