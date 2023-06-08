[Begin_ResourceLayout]

	cbuffer PerDrawCall : register(b0)
	{
		float4x4 WorldViewProj	: packoffset(c0);	[WorldViewProjection]
		float4x4 World			: packoffset(c4); 	[World]
	};

	cbuffer PerCamera : register(b1)
	{
		float3 CameraPosition		: packoffset(c0.x); [CameraPosition]
	}
	
	cbuffer Parameters : register(b2)
	{
		float3 LightPosition		: packoffset(c0.x); [Default(0,0,1)]
		float AmbientFactor 		: packoffset(c0.w); [Default(0.02)]
		float3 SSColor				: packoffset(c1.x); [Default(0.9, 0.26, 0.23)]
		float Metallic				: packoffset(c1.w);
		float Reflectance			: packoffset(c2.x); [Default(0.3)]
		float IrradiPerp 			: packoffset(c2.y); [Default(3)]
		float TDistortion 			: packoffset(c2.z); [Default(1)]
		float TPower				: packoffset(c2.w); [Default(1)]
		float TScale				: packoffset(c3.x); [Default(1)]
		float TAmbient				: packoffset(c3.y); [Default(0.02)]
	};

	Texture2D BaseTexture				: register(t0);
	Texture2D RoughnessAOThickness		: register(t1);
	Texture2D NormalTexture				: register(t2);
	TextureCube IBLIrradianceTexture	: register(t3); [IBLIrradiance]
	SamplerState TextureSampler			: register(s0);
	
[End_ResourceLayout]

[Begin_Pass:Default]
	[Profile 10_0]
	[Entrypoints VS=VS PS=PS]

	#define PI 3.14159265359f
	
	struct VS_IN
	{
		float4 position : POSITION;
		float3 normal	: NORMAL;
		float4 tangent	: TANGENT;
		float2 texCoord : TEXCOORD;
	};

	struct PS_IN
	{
		float4 position 	: SV_POSITION;
		float3 normal		: NORMAL0;
		float3 tangent		: TANGENT0;
		float3 bitangent	: BINORMAL0;
		float2 texCoord 	: TEXCOORD0;
		float3 positionWS 	: TEXCOORD1;
	};

	struct Surface
	{
		half3 albedo;
		half AO;
		half3 position;
		half thinkness;
		half3 normal;
		half reflectance;
		half3 viewVector;
		half metallic;
		half roughness;
		half NdotV;
		
		
		inline void Create( in half3 color,
							in half3 P,
							in half3 N,	
							in half3 viewPos,
							in half sAO,
							in half sthinkness,
							in half sroughness,
							in half smetallic,
							in half sreflectance)
		{
			albedo = color;
			position = P;
			normal = N;
			viewVector = normalize(viewPos - P);
			NdotV = saturate(dot(N, viewPos));
			AO = sAO;
			thinkness = sthinkness;
			metallic = smetallic;
			roughness = sroughness * sroughness;
			reflectance = sreflectance;
		}
	};
	
	half3 fresnelSchlick(half cosTheta, half3 F0)
	{
  		return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
	} 
	
	struct SurfaceToLight
	{
		half3 lightVector;
		half NdotL;
		half3 halfVector;
		half NdotH;
		half3 fresnelTerm;
		half VdotH;
		half irradiance;
		
		inline void Create(in Surface surface, in half3 lightDir)
		{
			lightVector = lightDir;
			halfVector = normalize(lightDir + surface.viewVector);
			NdotL = saturate(dot(lightVector, surface.normal));
			NdotH = saturate(dot(surface.normal, halfVector));
			VdotH = saturate(dot(surface.viewVector, halfVector));
			
			float3 f0 = 0.16 * (surface.reflectance * surface.reflectance);
			f0 = lerp(f0, surface.albedo, surface.metallic);
			fresnelTerm = fresnelSchlick(VdotH, f0);
			irradiance = max(dot(lightVector, surface.normal), 0.0) * IrradiPerp;
		}
	};

	PS_IN VS(VS_IN input)
	{
		PS_IN output = (PS_IN)0;

		output.position = mul(input.position, WorldViewProj);
		
		output.positionWS = mul(input.position, World).xyz;
		output.normal = mul(float4(input.normal, 0), World).xyz;
		output.tangent = mul(input.tangent, World).xyz;
		output.bitangent = cross(output.normal, output.tangent) * input.tangent.w;
		
		output.texCoord = input.texCoord;

		return output;
	}

	half3 GammaToLinear(in half3 color)
	{
		return pow(abs(color), 2.2);
	}
	
	half D_GGX(half NoH, half roughness)
	{
		half alpha = roughness * roughness;
		half alpha2 = alpha * alpha;
		half NoH2 = NoH * NoH;
		half b = (NoH2 * (alpha2 - 1.0) + 1.0);
		return alpha2 / (PI * b * b);
	}
	
	half G1_GGX_Schlick(half NdotV, half roughness)
	{
		//float r = roughness; // original
		half r = 0.5 + 0.5 * roughness; // Disney remapping
		half k = (r * r) / 2.0;
		half denom = NdotV * (1.0 - k) + k;
		return NdotV / denom;
	}
	
	half G_Smith(half NoV, half NoL, half roughness) 
	{
		half g1_l = G1_GGX_Schlick(NoL, roughness);
		half g1_v = G1_GGX_Schlick(NoV, roughness);
		return g1_l * g1_v;
	}
	
	half3 BRDFSpecular(in Surface surface, in SurfaceToLight surface2light)
	{
		half3 F = surface2light.fresnelTerm;
		half D = D_GGX(surface2light.NdotH, surface.roughness);
		half G = G_Smith(surface.NdotV, surface2light.NdotL, surface.roughness);
		
		half3 specular = (D * G * F) / max(4.0 * surface.NdotV * surface2light.NdotL, 0.001);
		
		return specular * surface2light.irradiance;
	}
	
	half3 BRDFDiffuse(in Surface surface, in SurfaceToLight surface2light)
	{
		half3 diffuseIrradiance = IBLIrradianceTexture.Sample(TextureSampler, surface.normal).rgb;
	
		half3 rhoD = 1.0 - surface2light.fresnelTerm; // if not specular, use as diffuse
		rhoD *= 1.0 - surface.metallic; // no diffuse for metals
		
		half3 ambient = surface.albedo * surface.AO * AmbientFactor * diffuseIrradiance;
		half3 diffuse = rhoD * surface.albedo / PI;
		
		return ambient + diffuse * surface2light.irradiance;
	}
	
	half3 ComputeTranslucency(in Surface surface, in SurfaceToLight surface2light)
	{
		half3 vLTLight = surface2light.lightVector + surface.normal * TDistortion;
		half fLTDot = pow(saturate(dot(surface.viewVector, -vLTLight)), TPower) * TScale;
		half3 fLT = (fLTDot + TAmbient) * surface.thinkness;
		return surface.albedo * fLT * SSColor;
	}

	float4 PS(PS_IN input) : SV_Target
	{
		half3 base = GammaToLinear(BaseTexture.Sample(TextureSampler, input.texCoord).rgb);		
		half3 RAT = RoughnessAOThickness.Sample(TextureSampler, input.texCoord).xyz;
				
		half3 normalTex = NormalTexture.Sample(TextureSampler, input.texCoord).rgb * 2 - 1;
		float3x3 tangentToWorld = float3x3(normalize(input.tangent), normalize(input.bitangent), normalize(input.normal));
		half3 normal = normalize(mul(normalTex, tangentToWorld));
		
		Surface surface;
		surface.Create(base,input.positionWS, normal, CameraPosition, RAT.y, RAT.z, RAT.x, Metallic, Reflectance);
		
		half3 lightDir = normalize(LightPosition - input.positionWS);
		SurfaceToLight surface2light;
		surface2light.Create(surface, lightDir);
		
		half3 diffuse = BRDFDiffuse(surface, surface2light);
		half3 specular = BRDFSpecular(surface, surface2light);
		half3 translucency = ComputeTranslucency(surface, surface2light);
		
		half3 radiance = diffuse + specular + translucency;
		
		return float4(radiance, 1.0);
	}

[End_Pass]