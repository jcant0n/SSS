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
		float ambientFactor 		: packoffset(c0.w); [Default(0.02)]
		float Metallic				: packoffset(c1.x);
		float Roughness				: packoffset(c1.y);
		float Reflectance			: packoffset(c1.z); [Default(0.3)]
		float irradiPerp 			: packoffset(c1.w); [Default(10)]
		float fLTDistortion			: packoffset(c2.x); [Default(1)]
		float iLTPower				: packoffset(c2.y); [Default(1)]
		float fLTScale				: packoffset(c2.z); [Default(1)]
		float fLTAmbient			: packoffset(c2.w); [Default(0.02)]
		float3 SSColor				: packoffset(c3.x); [Default(0.9, 0.26, 0.23)]
		float SubsurfaceScattering	: packoffset(c3.w); [Default(0.5)]
		float3 baseColor			: packoffset(c4.x); [Default(0.5, 0.19, 0.13)]
		float SubsurfaceRadius		: packoffset(c4.w); [Default(0.25)]
	};

	Texture2D BaseTexture				: register(t0);
	Texture2D RoughnessAOThickness		: register(t1);
	Texture2D NormalTexture				: register(t2);
	Texture2D RadianceTexture			: register(t3); [IBLRadiance]
	Texture2D IrradianceTexture			: register(t4); [IBLIrradiance]
	Texture2D BRDFIntegrationTexture	: register(t5); 
	SamplerState TextureSampler			: register(s0);
	
[End_ResourceLayout]

[Begin_Pass:Default]
	[Profile 10_0]
	[Entrypoints VS=VS PS=PS]

	#define PI 3.14159265359f
	#define RECIPROCAL_PI2 0.15915494f
	
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

	float2 DirectionToEquirectangular(float3 dir)
	{
		float lon = atan2(dir.z, dir.x);
		float lat = acos(dir.y);
		float2 sphereCoords = float2(lon, lat) * RECIPROCAL_PI2 * 2.0;
		float s = sphereCoords.x * 0.5 + 0.5;
		float t = sphereCoords.y;
		
		return float2(s, t);
	}

	// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
	float3 SpecularIBL(float3 F0 , float roughness, float3 N, float3 V)
	{
		float NoV = clamp(dot(N, V), 0.0, 1.0);
		float3 R = reflect(-V, N);
		float2 uv = DirectionToEquirectangular(R);
		float3 prefilteredColor = RadianceTexture.SampleLevel(TextureSampler, uv, roughness*float(6)).rgb; 
		float4 brdfIntegration = BRDFIntegrationTexture.Sample(TextureSampler, float2(NoV, roughness));
		return prefilteredColor * ( F0 * brdfIntegration.x + brdfIntegration.y );
	}

	float3 DiffuseIBL(float3 normal)
	{
		float2 uv = DirectionToEquirectangular(normal);
		return IrradianceTexture.Sample(TextureSampler, uv).rgb;
	}
	
	float3 fresnelSchlick(float cosTheta, float3 F0)
	{
  		return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
	} 
	
	float3 BRDFIBL(in float3 L, in float3 V, in float3 N, 
              in float3 baseColor, in float metallicness, in float roughness, in float fresnelReflect) 
  	{
		// F0 for dielectics in range [0.0, 0.16] 
		// default FO is (0.16 * 0.5^2) = 0.04
		float3 f0 = 0.16 * (fresnelReflect * fresnelReflect); 
		// in case of metals, baseColor contains F0
		f0 = lerp(f0, baseColor, metallicness);
		
		// compute diffuse and specular factors
		float NoV = clamp(dot(N, V), 0.0, 1.0);
		float3 F = fresnelSchlick(NoV, f0);
		float3 kS = F;
		float3 kD = 1.0 - kS;
		kD *= 1.0 - Metallic;
		
		float3 specular = SpecularIBL(f0, roughness, N, V); 
    	float3 diffuse = DiffuseIBL(N);
    
		// diffuse
		float3 color = kD * baseColor * diffuse + specular;
		
		return color;
	}

	// https://johnaustin.io/articles/2020/fast-subsurface-scattering-for-the-unity-urp
	half3 LightingSubsurface(float3 lightDir, half3 normalWS, half3 subsurfaceColor, half subsurfaceRadius)
	{
	    // Calculate normalized wrapped lighting. This spreads the light without adding energy.
	    // This is a normal lambertian lighting calculation (using N dot L), but warping NdotL
	    // to wrap the light further around an object.
	    //
	    // A normalization term is applied to make sure we do not add energy.
	    // http://www.cim.mcgill.ca/~derek/files/jgt_wrap.pdf
	
	    half NdotL = dot(normalWS, lightDir);
	    half alpha = subsurfaceRadius;
	    half theta_m = acos(-alpha); // boundary of the lighting function
	
	    half theta = max(0, NdotL + alpha) - alpha;
	    half normalization_jgt = (2 + alpha) / (2 * (1 + alpha));
	    half wrapped_jgt = (pow(((theta + alpha) / (1 + alpha)), 1 + alpha)) * normalization_jgt;
	
	    half wrapped_valve = 0.25 * (NdotL + 1) * (NdotL + 1);
	    half wrapped_simple = (NdotL + alpha) / (1 + alpha);
	
	    half3 subsurface_radiance = subsurfaceColor * wrapped_jgt;
	
	    return subsurface_radiance;
	}

	float4 PS(PS_IN input) : SV_Target
	{
		float3 base = BaseTexture.Sample(TextureSampler, input.texCoord).rgb;
		float3 RAT = RoughnessAOThickness.Sample(TextureSampler, input.texCoord).xyz;
		
		float3 normalTex = NormalTexture.Sample(TextureSampler, input.texCoord).rgb * 2 - 1;
		float3x3 tangentToWorld = float3x3(normalize(input.tangent), normalize(input.bitangent), normalize(input.normal));
		float3 normal = normalize(mul(normalTex, tangentToWorld));
		
		float3 viewDir = normalize(CameraPosition - input.positionWS);
		float3 lightDir = normalize(LightPosition - input.positionWS);
		float roughness = RAT.x;
		float metallic = Metallic; //mrTexture.y;
		float reflectance = Reflectance;
		
		float3 radiance = base * ambientFactor;
		float irradiance = max(dot(lightDir, normal), 0.0) * irradiPerp;
		//if(irradiance > 0.0) // if receives light
		//{
			float3 brdf = BRDFIBL(lightDir, viewDir, normal, base, metallic, roughness, reflectance);
			radiance += brdf * irradiance * RAT.y;// * lightColor.rgb;
		//}
		
		float3 vLTLight = lightDir + normal * fLTDistortion;
		float fLTDot = pow(saturate(dot(viewDir, -vLTLight)), iLTPower) * fLTScale;
		float3 fLT = (fLTDot + fLTAmbient) * RAT.z;
		radiance += base * fLT * SSColor;
		
		//float3 subsurfaceContribution = LightingSubsurface(lightDir, normal, SSColor, SubsurfaceRadius);
		//radiance = lerp(radiance, subsurfaceContribution, SubsurfaceScattering);

		return float4(radiance, 1.0);
	}

[End_Pass]