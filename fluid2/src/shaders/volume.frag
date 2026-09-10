#version 330 core
in vec2 uv;
out vec4 out_color;
uniform sampler3D volume;
uniform sampler3D velocity;
uniform mat4 inv_vp;
uniform vec3 eye;
uniform float exposure;
uniform float cut;
uniform vec2 value_range;
uniform int field;
uniform int beaming;
uniform int samples;
uniform int logarithmic;

vec3 palette(float t) {
    vec3 a=vec3(0.035,0.14,0.24), b=vec3(0.06,0.77,0.81);
    vec3 c=vec3(0.56,0.35,0.86), d=vec3(1.0,0.68,0.32);
    return t<0.38?mix(a,b,smoothstep(0.0,0.38,t)):t<0.7?mix(b,c,smoothstep(0.38,0.7,t)):mix(c,d,smoothstep(0.7,1.0,t));
}
vec3 tc(vec3 p){return p/vec3(3.0,2.0,2.0)+0.5;}
vec4 data(vec3 p){return texture(volume,tc(p));}
float value(vec4 s){return field==0?s.x:field==1?s.y:field==2?s.z:s.w;}
float norm(float a){return logarithmic==1?clamp(log(max(a,1e-8)/value_range.x)/log(value_range.y/value_range.x),0.0,1.0):clamp((a-value_range.x)/(value_range.y-value_range.x),0.0,1.0);}
vec2 box(vec3 ro,vec3 rd,vec3 bmin,vec3 bmax){vec3 t0=(bmin-ro)/rd,t1=(bmax-ro)/rd;vec3 a=min(t0,t1),b=max(t0,t1);return vec2(max(max(a.x,a.y),a.z),min(min(b.x,b.y),b.z));}
float hash(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
void main() {
    vec4 far=inv_vp*vec4(uv*2.0-1.0,1.0,1.0);
    vec3 rd=normalize(far.xyz/far.w-eye);
    vec3 bg=mix(vec3(0.0014,0.0028,0.0055),vec3(0.0035,0.007,0.012),clamp(1.0-length((uv-0.5)*1.4),0.0,1.0));
    // Laboratory floor with antialiased metric grid.
    float tf=(-1.07-eye.y)/rd.y;
    if(tf>0.0) {
        vec3 p=eye+rd*tf;
        vec2 q=p.xz*4.0;
        vec2 grid=abs(fract(q-0.5)-0.5)/max(fwidth(q),vec2(0.001));
        float line=1.0-min(min(grid.x,grid.y),1.0);
        float falloff=exp(-0.14*dot(p.xz,p.xz));
        bg+=vec3(0.004,0.011,0.016)*line*falloff;
        float ring=exp(-pow((length(p.xz)-2.12)*95.0,2.0));
        bg+=vec3(0.002,0.023,0.028)*ring;
        // The floor glow is a presentation aid, not radiative hydrodynamics.
        bg+=vec3(0.002,0.005,0.008)*exp(-dot(p.xz,p.xz)*0.8);
    }
    vec2 hit=box(eye,rd,vec3(-1.5,-1.0,-1.0),vec3(1.5,1.0,cut));
    vec3 col=vec3(0.0);float trans=1.0;
    if(hit.y>max(hit.x,0.0)) {
        float start=max(hit.x,0.0),step_size=(hit.y-start)/float(samples);
        float t=start+step_size*hash(gl_FragCoord.xy);
        for(int i=0;i<256;i++) {
            if(i>=samples||trans<0.008)break;
            vec3 p=eye+rd*t;vec4 s=data(p);
            float v=norm(value(s));
            vec3 gradient=vec3(
                norm(value(data(p+vec3(0.025,0,0))))-norm(value(data(p-vec3(0.025,0,0)))),
                norm(value(data(p+vec3(0,0.025,0))))-norm(value(data(p-vec3(0,0.025,0)))),
                norm(value(data(p+vec3(0,0,0.025))))-norm(value(data(p-vec3(0,0,0.025)))));
            float grad=length(gradient);
            float density=(0.08*v+1.0*pow(v,2.0)+8.5*grad)*exposure;
            vec3 base=palette(v);
            // Directional gradient illumination makes real density interfaces readable.
            float shade=0.72+0.55*abs(dot(gradient/(grad+0.001),normalize(vec3(-0.5,0.8,0.6))));
            float beam=1.0;
            if(beaming==1) {
                vec3 vel=texture(velocity,tc(p)).xyz;
                float delta=1.0/(s.w*(1.0-dot(vel,-rd)));
                beam=clamp(delta*delta*delta,0.025,15.0);
            }
            float alpha=1.0-exp(-density*step_size);
            col+=trans*alpha*base*shade*beam;
            trans*=1.0-alpha;
            t+=step_size;
        }
        // The cut surface is an actual sampled scalar field.
        if(cut<0.99 && abs(rd.z)>0.0001) {
            float plane=(cut-eye.z)/rd.z;vec3 p=eye+rd*plane;
            if(plane>0.0 && abs(p.x)<1.5 && abs(p.y)<1.0 && abs(plane-start)<0.02) {
                float v=norm(value(data(p)));float contour=1.0-smoothstep(0.0,0.04,abs(fract(v*12.0)-0.5));
                col=mix(col,palette(v)*(0.65+0.35*contour),0.73);trans*=0.27;
            }
        }
    }
    vec3 linear=col+trans*bg;
    linear*=1.0-0.2*pow(length(uv-0.5),2.0);
    out_color=vec4(linear,1.0);
}
